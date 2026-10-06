import argparse
import json
import multiprocessing as mp
import os
import re

import torch

from agents import game_specs
from memory_pathways import Pathways, build_agentic_prompt
from memory_pathways.load_model import load

CONDITIONS = {"full": "both", "attention_only": "attention", "recurrent_only": "recurrent"}
N_GAMES = {"textworld": 200, "babyai": 100}


# ------------------------------------------------------------------ env process
def _serve(spec, conn):
    from agents import make_env
    if spec["env"] == "textworld":
        from agents.textworld_env import workdir
        workdir()
    env = make_env(spec)
    first = env.reset()
    conn.send((first, env.max_steps, env.parse_last, env.max_new_tokens))
    while True:
        action = conn.recv()
        if action == "__close__":
            env.close()
            return
        conn.send(env.step(action))


class RemoteEnv:
    def __init__(self, spec, timeout):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(spec, child), daemon=True)
        self.proc.start()
        self.timeout = timeout
        self.first, self.max_steps, self.parse_last, self.max_new_tokens = self._recv()

    def _recv(self):
        if not self.conn.poll(self.timeout):
            raise TimeoutError("environment did not respond")
        return self.conn.recv()

    def step(self, action):
        self.conn.send(action)
        return self._recv()

    def close(self):
        try:
            self.conn.send("__close__")
            self.proc.join(10)
        finally:
            if self.proc.is_alive():
                self.proc.kill()


def parse_action(text, last):
    cands = [c.strip() for c in re.findall(r"<action>(.*?)</action>", text or "", re.DOTALL | re.IGNORECASE)]
    cands = [c for c in cands if c]
    return (cands[-1] if last else cands[0]) if cands else None


# ------------------------------------------------------------------ episode
def play(model, tok, family, paths, mode, spec, a):
    env = RemoteEnv(spec, a.env_timeout)
    try:
        messages = [{"role": "user", "content": env.first}]
        turns, won = [], False
        dev = next(model.parameters()).device
        for _ in range(env.max_steps):
            p = build_agentic_prompt(tok, messages, family)
            ids, segs = p["ids"], p["segs"]
            budget = a.max_len - env.max_new_tokens
            if len(ids) > budget:                     # keep the task description and the most recent turns
                keep = list(range(1024)) + list(range(len(ids) - (budget - 1024), len(ids)))
                ids, segs = [ids[i] for i in keep], [segs[i] for i in keep]
            ids_t = torch.tensor([ids], device=dev)
            with torch.no_grad(), paths.restrict(mode, torch.tensor(segs), a.window):
                out = model.generate(ids_t, max_new_tokens=env.max_new_tokens, do_sample=True, temperature=0.4,
                                     top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id,
                                     stop_strings=None if env.parse_last else ["</action>"], tokenizer=tok)
            gen = tok.decode(out[0, ids_t.shape[1]:], skip_special_tokens=True).strip()
            action = parse_action(gen, env.parse_last)
            obs, won, done, status = env.step(action)
            turns.append({"generation": gen, "action": action, "status": status})
            messages += [{"role": "assistant", "content": gen}, {"role": "user", "content": obs}]
            if won or done:
                break
        return {"id": spec["id"], "won": won, "steps": len(turns), "turns": turns}
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["textworld", "babyai"])
    ap.add_argument("--game", required=True, help="textworld: quest | treasure; babyai: level name")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n", type=int, default=None, help="games (default 200 TextWorld / 100 BabyAI)")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--condition", default="full", choices=list(CONDITIONS))
    ap.add_argument("--window", type=int, default=1, help="recurrent_only: turns visible to attention")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=None, help="context budget (default 65536 TextWorld / 32768 BabyAI)")
    ap.add_argument("--env_timeout", type=int, default=300, help="seconds to wait for the environment")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    a.max_len = a.max_len or (65536 if a.env == "textworld" else 32768)
    specs = game_specs(a.env, a.game, a.split, a.n or N_GAMES[a.env])[a.shard::a.n_shards]
    done = {}
    if os.path.exists(a.out):                     # resume
        done = {r["id"]: r for r in json.load(open(a.out))["games"]}

    model, tok, family = load(a.model)
    if a.ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.ckpt).merge_and_unload()
    model.eval()
    paths = Pathways(model)
    mode = CONDITIONS[a.condition]

    def save():
        games = list(done.values())
        ok = [g for g in games if "error" not in g]
        summary = {"success_rate": 100 * sum(g["won"] for g in ok) / max(1, len(ok)), "n": len(ok),
                   "skipped": len(games) - len(ok),
                   "avg_steps_won": sum(g["steps"] for g in ok if g["won"]) / max(1, sum(g["won"] for g in ok))}
        tmp = a.out + ".partial"
        json.dump({"config": vars(a), "summary": summary, "games": games}, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)
        return summary

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for i, spec in enumerate(specs):
        if spec["id"] in done:
            continue
        torch.manual_seed(a.seed * 100003 + spec["seed"])
        try:
            done[spec["id"]] = play(model, tok, family, paths, mode, spec, a)
        except Exception as e:                      # generation failure or hung game: excluded from the rate
            print(f"[skip] {spec['id']}: {e!r}", flush=True)
            done[spec["id"]] = {"id": spec["id"], "error": repr(e)}
        s = save()
        print(f"[{i + 1}/{len(specs)}] {spec['id']} won={done[spec['id']].get('won')} "
              f"| success {s['success_rate']:.1f}% (n={s['n']})", flush=True)
    print(json.dumps(save()))


if __name__ == "__main__":
    main()

import argparse
import json
import multiprocessing as mp
import os
import signal

from agents import game_specs, make_env

MAX_GOLD_STEPS = {"textworld": 40}


def _init():
    if os.environ.get("BMP_ENV") == "textworld":
        from agents.textworld_env import workdir
        workdir()


def _timeout(*_):
    raise TimeoutError


def rollout(spec, timeout=300):
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout)
    env = None
    try:
        env = make_env(spec)
        messages = [{"role": "user", "content": env.reset()}]
        cap = MAX_GOLD_STEPS.get(spec["env"], env.max_steps)
        for _ in range(cap):
            a = env.gold_action()
            if a is None:
                break
            messages.append({"role": "assistant", "content": f"<action>{a}</action>"})
            nxt, won, done, _ = env.step(a)
            if won:
                return {"id": spec["id"], "messages": messages,
                        "n_actions": sum(m["role"] == "assistant" for m in messages)}
            if done:
                break
            messages.append({"role": "user", "content": nxt})
        return None
    except Exception as e:
        print(f"[skip] {spec['id']}: {e!r}", flush=True)
        return None
    finally:
        signal.alarm(0)
        if env is not None:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["textworld", "babyai"])
    ap.add_argument("--game", required=True, help="textworld: quest | treasure; babyai: level name")
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, required=True, help="number of trajectories to collect")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=None, help="seconds per game (default 300 TW / 5 BabyAI)")
    a = ap.parse_args()
    timeout = a.timeout or (300 if a.env == "textworld" else 5)
    os.environ["BMP_ENV"] = a.env
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # unsolved games are dropped; BabyAI draws further seeds of the same split until n are collected
    specs = game_specs(a.env, a.game, a.split, a.n if a.env == "textworld" else 3 * a.n)
    kept = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, maxtasksperchild=20) as pool, open(out, "w") as f:
        for r in pool.imap(_call, [(s, timeout) for s in specs]):
            if r is None or len(kept) >= a.n:
                continue
            kept.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out}: {len(kept)} trajectories from {len(specs)} games "
          f"(mean {sum(r['n_actions'] for r in kept) / max(1, len(kept)):.1f} actions)")


def _call(args):
    return rollout(*args)


if __name__ == "__main__":
    main()

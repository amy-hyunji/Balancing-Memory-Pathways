import argparse
import json
import math
import os
import random

import torch

from memory_pathways import OBJECTIVES, Pathways, build_agentic, build_qa, objective_backward, to_tensors
from memory_pathways.load_model import LORA_TARGETS, MODELS, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b", help=f"{list(MODELS)} or a HF id")
    ap.add_argument("--task", required=True, choices=["qa", "agentic"])
    ap.add_argument("--objective", required=True, choices=list(OBJECTIVES))
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.25, help="weight of the auxiliary term")
    ap.add_argument("--window", type=int, default=1,
                    help="recurrent-only: attention sees the last `window` segments")
    ap.add_argument("--qa_boundary", default="question", choices=["response", "question"],
                    help="QA segment boundary: before the assistant response, or before the question")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch_size", type=int, default=16, help="effective batch (sequences per step)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3, help="evaluations without improvement before stopping")
    ap.add_argument("--max_len", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from peft import LoraConfig, get_peft_model

    model, tok, family = load(a.model)
    dev = next(model.parameters()).device

    def read(path):
        out, bad = [], 0
        for line in open(path):
            r = json.loads(line)
            try:
                ex = build_qa(tok, r, family, a.qa_boundary) if a.task == "qa" else build_agentic(tok, r, family)
            except ValueError as e:
                bad += 1
                print(f"[data] drop row: {e}", flush=True)
                continue
            if len(ex["ids"]) > a.max_len:
                bad += 1
                continue
            out.append(ex)
        return out, bad

    train, bad = read(a.train)
    valid, _ = read(a.valid)
    print(f"[data] train {len(train)} (dropped {bad}: template mismatch or > --max_len) | valid {len(valid)}", flush=True)

    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS[family]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    paths = Pathways(model)

    # sanity: each restriction must change the prediction right after the first boundary
    ids, _, segs = to_tensors(next(e for e in train if max(e["segs"]) > 0), dev)
    p = segs.tolist().index(1) + 2
    with torch.no_grad():
        ref = model(input_ids=ids[:, :p], use_cache=False).logits[0, -1].float()
        for mode in ("recurrent", "attention"):
            with paths.restrict(mode, segs[:p], a.window):
                d = (model(input_ids=ids[:, :p], use_cache=False).logits[0, -1].float() - ref).abs().max()
            print(f"[sanity] {mode}-only max|Δlogit| = {float(d):.4f}", flush=True)
            assert d > 1e-3, f"{mode}-only restriction had no effect"

    n_forward = len(OBJECTIVES[a.objective])
    steps = max(1, math.ceil(a.epochs * len(train) / a.batch_size / n_forward))
    print(f"[train] {a.objective}: {steps} steps ({n_forward} forward(s) per example)", flush=True)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(steps * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))))

    def order():
        e = 0
        while True:
            idx = list(range(len(train)))
            random.Random(a.seed * 1000 + e).shuffle(idx)
            yield from idx
            e += 1
    it = order()

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0
        for ex in valid:
            ids, lab, segs = to_tensors(ex, dev)
            tot += objective_backward(model, paths, a.objective, ids, lab, segs, a.lam, a.window, backward=False)[1]
        model.train()
        return tot / max(1, len(valid))

    def save(path, step, val):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        json.dump({**vars(a), "step": step, "valid_loss": val}, open(os.path.join(path, "train_meta.json"), "w"),
                  indent=2)

    best, bad_evals = float("inf"), 0
    model.train()
    opt.zero_grad()
    for step in range(1, steps + 1):
        logs = {}
        for _ in range(a.batch_size):
            ids, lab, segs = to_tensors(train[next(it)], dev)
            for k, v in objective_backward(model, paths, a.objective, ids, lab, segs, a.lam, a.window,
                                           scale=1.0 / a.batch_size)[0].items():
                logs.setdefault(k, []).append(v)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 5 == 0:
            msg = " ".join(f"{k} {sum(v) / len(v):.4f}" for k, v in logs.items())
            print(f"step {step}/{steps} {msg} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % a.eval_every == 0 or step == steps:
            val = evaluate()
            print(f"[eval] step {step} valid {val:.4f} (best {best:.4f})", flush=True)
            if val < best:
                best, bad_evals = val, 0
                save(os.path.join(a.out, "best"), step, val)
            else:
                bad_evals += 1
                if bad_evals >= a.patience:
                    print(f"[early stop] step {step}", flush=True)
                    break
    print(f"[done] best valid {best:.4f} -> {os.path.join(a.out, 'best')}", flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
import os

import torch

from memory_pathways import Pathways, build_qa_prompt
from memory_pathways.load_model import load

MODES = {"full": "both", "attention_only": "attention", "recurrent_only": "recurrent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa_boundary", default="question", choices=["response", "question"])
    ap.add_argument("--max_new", type=int, default=64)
    a = ap.parse_args()

    model, tok, family = load(a.model)
    if a.ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.ckpt)
    model.eval()
    paths = Pathways(model)
    dev = next(model.parameters()).device

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for line in open(a.data):
            r = json.loads(line)
            p = build_qa_prompt(tok, r, family, a.qa_boundary)
            ids, segs = torch.tensor([p["ids"]], device=dev), torch.tensor(p["segs"])
            gen = {}
            for name, mode in MODES.items():
                with torch.no_grad(), paths.restrict(mode, segs):
                    out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                gen[name] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({k: r.get(k) for k in ("id", "dataset", "metric", "golds")} | gen,
                               ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

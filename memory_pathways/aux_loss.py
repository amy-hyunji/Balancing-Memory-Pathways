import torch.nn.functional as F

# objective -> list of (restriction, weight); weight None = lambda
OBJECTIVES = {
    "sft":      [("both", 1.0)],
    "sft+rec":  [("both", 1.0), ("recurrent", None)],
    "rec":      [("recurrent", 1.0)],
    "attn":     [("attention", 1.0)],
    "sft+attn": [("both", 1.0), ("attention", None)],
}
NAMES = {"both": "L_SFT", "recurrent": "L_rec", "attention": "L_attn"}


def ce(model, ids, labels):
    """Mean next-token CE over supervised positions; logits are computed only
    there (full logits over a large vocabulary and long context do not fit)."""
    pos = (labels[0, 1:] != -100).nonzero().squeeze(1)
    out = model(input_ids=ids, logits_to_keep=pos, use_cache=False)
    return F.cross_entropy(out.logits[0].float(), labels[0, pos + 1])


def objective_backward(model, paths, objective, ids, labels, segs, lam=0.25, window=1, scale=1.0,
                       backward=True):
    """Runs every term of `objective` (each with its own backward unless
    backward=False); returns {term: loss} and the weighted total."""
    logs, total = {}, 0.0
    for mode, w in OBJECTIVES[objective]:
        w = lam if w is None else w
        with paths.restrict(mode, segs, window):
            l = ce(model, ids, labels)
            if backward:
                (w * scale * l).backward()
        logs[NAMES[mode]] = float(l)
        total += w * float(l)
    return logs, total

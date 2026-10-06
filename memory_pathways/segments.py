import torch

TURN_START = {"qwen3_5": "<|im_start|>", "nemotron_h": "<SPECIAL_11>"}


def _enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids if s else []


def _render(tok, messages, add_generation_prompt=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


def _encode(tok, full, seg_cuts, lab_spans):
    """Tokenize `full` piecewise at every boundary; label the characters inside
    lab_spans; segment id = number of seg_cuts <= piece start."""
    bounds = sorted({0, len(full), *seg_cuts, *[x for s in lab_spans for x in s]})
    ids, labels, segs = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t = _enc(tok, full[a:b])
        on = any(x <= a and b <= y for x, y in lab_spans)
        ids += t
        labels += t if on else [-100] * len(t)
        segs += [sum(c <= a for c in seg_cuts)] * len(t)
    if ids != _enc(tok, full):
        raise ValueError("piecewise encoding != whole-string encoding")
    if lab_spans and lab_spans[-1][1] == len(full) and ids[-1] != tok.eos_token_id:
        # the template does not close the last assistant turn (Nemotron-H): close it
        # the way it closes earlier turns, "\n" + end-of-turn
        t = _enc(tok, "\n" + tok.eos_token)
        ids += t
        labels += t
        segs += [segs[-1]] * len(t)
    return ids, labels, segs


def _locate(tok, full, msgs, family, cut_role, cut_first):
    """Character spans of supervised assistant messages, and segment cuts for
    every `cut_role` message (the first one only if cut_first):
      assistant : at the assistant header (the response is cut from its prompt)
      user      : right after the previous assistant turn's end-of-turn token,
                  so a turn is [observation ... action + end-of-turn]"""
    turn = TURN_START[family]
    eos = tok.eos_token or ""
    cur, cuts, spans, seen = 0, [], [], False
    for m in msgs:
        j = full.find(m["content"], cur)
        if j < 0:
            raise ValueError("message content not found in rendered template")
        if m["role"] == cut_role:
            if seen or cut_first:
                # (the header may already be consumed as the previous turn's end-of-turn token)
                cuts.append(cur if cut_role == "user" else max(full.rfind(turn, 0, j), cur))
            seen = True
        end = j + len(m["content"])
        if m["role"] == "assistant":
            while end < len(full) and full[end] in " \t\n" and not (eos and full.startswith(eos, end)):
                end += 1
            if eos and full.startswith(eos, end):
                end += len(eos)                   # the model must learn to stop
            if m.get("train", True):
                spans.append((j, end))
        cur = end
    return cuts, spans


def build_agentic(tok, row, family):
    msgs = [{**m, "content": m["content"].strip()} for m in row["messages"]]
    full = _render(tok, [{"role": m["role"], "content": m["content"]} for m in msgs])
    cuts, spans = _locate(tok, full, msgs, family, "user", cut_first=False)
    ids, labels, segs = _encode(tok, full, cuts, spans)
    return {"ids": ids, "labels": labels, "segs": segs}


def build_agentic_prompt(tok, messages, family):
    """Generation prompt after a trajectory that ends with a user turn, with the
    same turn segments as in training; the generated action belongs to the last."""
    msgs = [{**m, "content": m["content"].strip()} for m in messages]
    full = _render(tok, [{"role": m["role"], "content": m["content"]} for m in msgs], add_generation_prompt=True)
    cuts, _ = _locate(tok, full, msgs, family, "user", cut_first=False)
    ids, _, segs = _encode(tok, full, cuts, [])
    return {"ids": ids, "segs": segs}


def _qa_messages(row):
    return [{"role": "user", "content": row["context"].strip() + "\n" + row["question"].strip()},
            {"role": "assistant", "content": str(row["answer"]).strip()}]


def build_qa(tok, row, family, boundary="question"):
    msgs = _qa_messages(row)
    full = _render(tok, msgs)
    cuts, spans = _locate(tok, full, msgs, family, "assistant", cut_first=True)
    if boundary == "question":
        q = row["question"].strip()
        cuts = [full.rfind(q, 0, cuts[0])]
    elif boundary != "response":
        raise ValueError(boundary)
    ids, labels, segs = _encode(tok, full, cuts, spans)
    return {"ids": ids, "labels": labels, "segs": segs}


def build_qa_prompt(tok, row, family, boundary="question"):
    """Generation prompt with the same segments as in training."""
    msgs = _qa_messages(row)[:1]
    full = _render(tok, msgs, add_generation_prompt=True)
    cut = full.rfind(TURN_START[family])
    if boundary == "question":
        cut = full.rfind(row["question"].strip(), 0, cut)
    ids = _enc(tok, full[:cut]) + _enc(tok, full[cut:])
    if ids != _enc(tok, full):
        raise ValueError("piecewise encoding != whole-string encoding")
    n0 = len(_enc(tok, full[:cut]))
    return {"ids": ids, "segs": [0] * n0 + [1] * (len(ids) - n0)}


def to_tensors(ex, device):
    return (torch.tensor([ex["ids"]], device=device),
            torch.tensor([ex["labels"]], device=device),
            torch.tensor(ex["segs"]))

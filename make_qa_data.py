import argparse
import ast
import json
import os
import random
import re

from datasets import load_dataset

INSTRUCTION = "Answer with a short answer only (a few words), no explanation, in the form: Answer: <answer>"
METRIC = {"drop": "em", "hotpotqa": "em", "qasper": "f1", "narrativeqa": "f1", "minteval": "em", "musr": "em"}
TRAIN_SETS = ["drop", "hotpotqa", "qasper", "narrativeqa"]
TEST_CAP = {"drop": 1500, "hotpotqa": 1500, "narrativeqa": 1500, "minteval": 1500}


def make(ds, rid, context, question, golds):
    golds = [g for g in dict.fromkeys(str(g).strip() for g in golds) if g]
    if not golds or not context.strip():
        return None
    return {"id": f"{ds}_{rid}", "dataset": ds, "context": context.strip(),
            "question": question.strip() + "\n\n" + INSTRUCTION, "answer": "Answer: " + golds[0],
            "golds": golds, "metric": METRIC[ds]}


# ---------------------------------------------------------------- loaders
def drop(split):
    for r in load_dataset("ucinlp/drop", split=split):
        yield make("drop", r["query_id"], r["passage"], r["question"], r["answers_spans"]["spans"])


def hotpotqa(split):
    for r in load_dataset("hotpotqa/hotpot_qa", "distractor", split=split):
        ctx = "\n\n".join(f"[{t}] {''.join(s)}" for t, s in zip(r["context"]["title"], r["context"]["sentences"]))
        yield make("hotpotqa", r["id"], ctx, r["question"], [r["answer"]])


def qasper(split):
    """Full paper (section names and paragraphs); one gold per annotator; unanswerable questions dropped."""
    for p in load_dataset("allenai/qasper", revision="refs/convert/parquet", split=split):
        ft = p["full_text"]
        ctx = "\n".join(x for nm, pl in zip(ft["section_name"], ft["paragraphs"]) for x in ([nm] if nm else []) + list(pl or []))
        for qid, q, ans in zip(p["qas"]["question_id"], p["qas"]["question"], p["qas"]["answers"]):
            golds = []
            for an in ans["answer"]:
                if an.get("unanswerable"):
                    continue
                if an.get("free_form_answer"):
                    golds.append(an["free_form_answer"])
                elif an.get("extractive_spans"):
                    golds.append(" ".join(an["extractive_spans"]))
                elif an.get("yes_no") is not None:
                    golds.append("Yes" if an["yes_no"] else "No")
            yield make("qasper", qid, ctx, q, golds)


def _gutenberg(t):
    t = t.replace("\ufeff", "")
    m = re.search(r"\*\*\*\s*START OF TH[EI]S? PROJECT GUTENBERG.*?\*\*\*", t, re.S)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\*\s*END OF TH[EI]S? PROJECT GUTENBERG", t, re.S)
    if m:
        t = t[:m.start()]
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def narrativeqa(split):
    """Full story (not the summary)."""
    docs = {}
    for i, r in enumerate(load_dataset("deepmind/narrativeqa", split=split, streaming=True)):
        d = r["document"]
        if d["id"] not in docs:
            docs[d["id"]] = _gutenberg(d["text"])
        yield make("narrativeqa", f"{d['id']}_{i}", docs[d["id"]], r["question"]["text"],
                   [a["text"] for a in r["answers"]])


def minteval(_split=None):
    for sub in ("wiki_revisions", "github_commits"):
        for r in load_dataset("dinobby/MINTEval", split=sub):
            ctx = "\n".join(f"[{e.get('timestamp', '')}] {e.get('content', '')}".strip() for e in r["contexts"])
            for k, q in enumerate(r["questions"]):
                q = json.loads(q) if isinstance(q, str) else q
                yield make("minteval", f"{sub}_{r['id']}_{k}", ctx, q["question"], [q["answer"]])


def musr(_split=None):
    for sub in ("murder_mysteries", "object_placements", "team_allocation"):
        for i, r in enumerate(load_dataset("TAUR-Lab/MuSR", split=sub)):
            choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
            q = r["question"] + "\nOptions: " + " | ".join(map(str, choices))
            yield make("musr", f"{sub}_{i}", r["narrative"], q, [choices[r["answer_index"]]])


LOAD = {"drop": drop, "hotpotqa": hotpotqa, "qasper": qasper, "narrativeqa": narrativeqa,
        "minteval": minteval, "musr": musr}
TEST_SPLIT = {"drop": "validation", "hotpotqa": "validation", "qasper": "test", "narrativeqa": "test",
              "minteval": None, "musr": None}


# ---------------------------------------------------------------- assembly
def quotas(avail, total):
    """Equal share per dataset; a dataset with fewer examples passes its shortfall to the others."""
    q, left, open_ = {k: 0 for k in avail}, total, set(avail)
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in sorted(open_):
            add = min(share, avail[k] - q[k], left)
            q[k] += add
            left -= add
        open_ = {k for k in open_ if q[k] < avail[k]}
    return q


def split_by_document(rows, n_valid, rng):
    """Whole documents go to valid until it holds n_valid questions."""
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["context"], []).append(r)
    docs = list(by_doc)
    rng.shuffle(docs)
    valid, train = [], []
    for d in docs:
        (valid if len(valid) < n_valid else train).extend(by_doc[d])
    return train, valid


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)}", {d: sum(r['dataset'] == d for r in rows) for d in dict.fromkeys(r['dataset'] for r in rows)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/qa")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_valid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # train / valid from the official training splits
    pools = {}
    for ds in TRAIN_SETS:
        rows = [r for r in LOAD[ds]("train") if r]
        tr, va = split_by_document(rows, a.n_valid // len(TRAIN_SETS), random.Random(a.seed))
        pools[ds] = (tr, va)
        print(f"[{ds}] train pool {len(tr)} | valid pool {len(va)}", flush=True)
    for name, idx, n in (("train", 0, a.n_train), ("valid", 1, a.n_valid)):
        q = quotas({ds: len(pools[ds][idx]) for ds in TRAIN_SETS}, n)
        rows = [r for ds in TRAIN_SETS for r in random.Random(a.seed).sample(pools[ds][idx], q[ds])]
        random.Random(a.seed).shuffle(rows)
        write(os.path.join(a.out, f"{name}.jsonl"), rows)

    # test: each dataset separately
    test = []
    for ds in ["drop", "musr", "hotpotqa", "qasper", "narrativeqa", "minteval"]:
        rows = [r for r in LOAD[ds](TEST_SPLIT[ds]) if r]
        if ds in TEST_CAP and len(rows) > TEST_CAP[ds]:
            rows = random.Random(a.seed).sample(rows, TEST_CAP[ds])
        test += rows
    write(os.path.join(a.out, "test.jsonl"), test)


if __name__ == "__main__":
    main()

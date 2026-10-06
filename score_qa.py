import json
import re
import string
import sys
from collections import Counter, defaultdict

ORDER = ["drop", "musr", "hotpotqa", "qasper", "narrativeqa", "minteval"]
_ANS = re.compile(r"(?m)^[ \t]*(?:\*\*)?Answer:(?:\*\*)?[ \t]*(.*)$")


def extract(text):
    m = _ANS.search(text or "")
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in (text or "").strip().split("\n") if l.strip()]
    return lines[0] if lines else ""


def normalize(s):
    s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def em(pred, golds):
    return float(any(normalize(pred) == normalize(g) for g in golds))


def f1(pred, golds):
    def one(g):
        p, t = normalize(pred).split(), normalize(g).split()
        if not p or not t:
            return float(p == t)
        common = sum((Counter(p) & Counter(t)).values())
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(t)
        return 2 * prec * rec / (prec + rec)
    return max(one(g) for g in golds)


def main(path):
    rows = [json.loads(l) for l in open(path)]
    conds = [c for c in ("full", "attention_only", "recurrent_only") if c in rows[0]]
    for c in conds:
        per, allv = defaultdict(list), []
        for r in rows:
            s = (f1 if r["metric"] == "f1" else em)(extract(r[c]), r["golds"])
            per[r["dataset"]].append(s)
            allv.append(s)
        cols = [d for d in ORDER if d in per]
        print(f"{c:15s} " + " ".join(f"{d} {100 * sum(per[d]) / len(per[d]):.1f}" for d in cols)
              + f" | avg {100 * sum(allv) / len(allv):.1f} (n={len(allv)})")


if __name__ == "__main__":
    main(sys.argv[1])

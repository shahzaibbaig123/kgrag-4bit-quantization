"""
Lexical/semantic metrics on EXISTING eval answers (no re-generation).

For verbose RAG answers vs short SQuAD gold spans, precision-based metrics
(EM/BLEU/ROUGE-F1) unfairly punish verbosity, so we report recall-style
"is the gold answer present?" metrics, taking the max over ALL gold answers:

  - Cover-EM      : normalized gold span is a substring of the answer (0/1)
  - Token-recall  : SQuAD-normalized token overlap recall (gold covered by answer)
  - BERTScore-rec : semantic/paraphrase-aware recall (optional, --bertscore)

Usage:
  python lexical_eval.py --size 200
  python lexical_eval.py --size 1000 --bertscore        # adds BERTScore recall
"""
import os
import re
import json
import string
import argparse
from collections import Counter

RESULTS_DIR = "./SQUAD/results"
DATA_DIR = "./data"
MODELS = {"f16": "qwenf16", "q4": "qwenQ4"}  # index model -> result filename tag

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    """SQuAD answer normalization: lowercase, drop punctuation + articles, squeeze spaces."""
    s = (s or "").lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_results(path):
    """Parse #/Q/Gold/Answer blocks from an eval result .txt file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pat = re.compile(
        r"#\d+\nQ:\s*(.*?)\nGold:\s*(.*?)\nAnswer:\s*(.*?)\n(?:Score:.*?\n)?-{20,}",
        re.DOTALL,
    )
    out = []
    for m in pat.finditer(text):
        out.append({"question": m.group(1).strip(),
                    "gold": m.group(2).strip(),
                    "answer": m.group(3).strip()})
    return out


def load_all_golds(size, version):
    """question -> list of ALL gold answers (SQuAD can have several)."""
    path = f"{DATA_DIR}/squad_train_{size}_v{version}.jsonl"
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m[d["question"].strip()] = [g for g in d["answers"]["text"] if g.strip()]
    return m


def cover_em(answer, golds):
    a = normalize(answer)
    return 1.0 if any(normalize(g) in a for g in golds if normalize(g)) else 0.0


def token_recall(answer, golds):
    a_tok = Counter(normalize(answer).split())
    best = 0.0
    for g in golds:
        g_tok = normalize(g).split()
        if not g_tok:
            continue
        same = sum((Counter(g_tok) & a_tok).values())
        best = max(best, same / len(g_tok))
    return best


def eval_one(size, model_tag, use_bertscore, bs_scorer):
    """Returns per-version dicts for one (size, model)."""
    per_version = []
    for v in range(5):
        rpath = os.path.join(RESULTS_DIR, f"eval_squad_{size}_v{v}_{model_tag}.txt")
        if not os.path.exists(rpath):
            continue
        samples = parse_results(rpath)
        golds_map = load_all_golds(size, v)

        cem, trec = [], []
        bs_pairs_cand, bs_pairs_ref = [], []   # for batched BERTScore (best gold = gold[0] here)
        for s in samples:
            golds = golds_map.get(s["question"], [s["gold"]]) or [s["gold"]]
            cem.append(cover_em(s["answer"], golds))
            trec.append(token_recall(s["answer"], golds))
            if use_bertscore:
                bs_pairs_cand.append(s["answer"])
                bs_pairs_ref.append(golds[0])  # train usually has 1 gold

        rec = {"version": v, "n": len(samples),
               "cover_em": sum(cem) / len(cem) if cem else 0.0,
               "token_recall": sum(trec) / len(trec) if trec else 0.0}

        if use_bertscore and bs_pairs_cand:
            P, R, F = bs_scorer(bs_pairs_cand, bs_pairs_ref, verbose=False, batch_size=64)
            rec["bertscore_recall"] = float(R.mean())
        per_version.append(rec)
    return per_version


def summarize(size, model_tag, per_version, use_bertscore):
    if not per_version:
        print(f"  (no result files for {model_tag} size {size})")
        return
    import statistics as st
    keys = ["cover_em", "token_recall"] + (["bertscore_recall"] if use_bertscore else [])
    out = os.path.join(RESULTS_DIR, f"lexical_{size}_{model_tag}_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"=== LEXICAL/SEMANTIC METRICS: SQuAD size={size}, model={model_tag} ===\n")
        f.write(f"Seeds present: {len(per_version)}/5\n\n")
        for r in per_version:
            line = f"v{r['version']} (n={r['n']}): " + "  ".join(f"{k}={r[k]:.4f}" for k in keys)
            f.write(line + "\n")
        f.write("\n")
        for k in keys:
            vals = [r[k] for r in per_version]
            f.write(f"Mean {k}: {st.mean(vals):.4f} (std {st.pstdev(vals):.4f})\n")
    print(f"  saved {out}")
    print("   " + " | ".join(
        f"{k}={st.mean([r[k] for r in per_version]):.4f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--bertscore", action="store_true", help="also compute BERTScore recall (slow; needs bert_score)")
    args = ap.parse_args()

    bs_scorer = None
    if args.bertscore:
        from bert_score import score as _bs
        bs_scorer = lambda cands, refs, **kw: _bs(cands, refs, lang="en", **kw)

    for model_dir, tag in MODELS.items():
        print(f"== model={tag}, size={args.size} ==")
        pv = eval_one(args.size, tag, args.bertscore, bs_scorer)
        summarize(args.size, tag, pv, args.bertscore)


if __name__ == "__main__":
    main()

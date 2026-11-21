"""
Lexical metrics on EXISTING TriviaQA eval answers (offline, no re-generation).

Reuses normalize/cover_em/token_recall from lexical_eval.py. Golds for each
question = answers.value + answers.aliases (max over all). Single seed, per size.

Usage:  python lexical_eval_triviaqa.py [--size 25]
"""
import os, json, argparse
import statistics as st

from lexical_eval import normalize, cover_em, token_recall, parse_results  # reuse read-only

RESULTS_DIR = "./TRIVIAQA/results"
DATA_DIR = "./data"
MODELS = {"f16": "qwenf16", "q4": "qwenQ4"}


def load_all_golds(size):
    """question -> [value + aliases] (deduped, non-empty)."""
    path = f"{DATA_DIR}/triviaqa_docs_{size}.jsonl"
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = d.get("answers") or {}
            golds = [ans.get("value", "")] + list(ans.get("aliases") or [])
            m[d["question"].strip()] = [g for g in dict.fromkeys(golds) if g and g.strip()]
    return m


def eval_one(size, model_tag):
    rpath = os.path.join(RESULTS_DIR, f"eval_triviaqa_{size}_{model_tag}.txt")
    if not os.path.exists(rpath):
        return None
    samples = parse_results(rpath)
    golds_map = load_all_golds(size)
    cem, trec = [], []
    for s in samples:
        golds = golds_map.get(s["question"], [s["gold"]]) or [s["gold"]]
        cem.append(cover_em(s["answer"], golds))
        trec.append(token_recall(s["answer"], golds))
    if not samples:
        return None
    return {"size": size, "n": len(samples),
            "cover_em": sum(cem) / len(cem), "token_recall": sum(trec) / len(trec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, choices=[25, 50, 75, 100])
    args = ap.parse_args()
    sizes = [args.size] if args.size else [25, 50, 75, 100]

    for model_dir, tag in MODELS.items():
        rows = [r for r in (eval_one(s, tag) for s in sizes) if r]
        if not rows:
            print(f"== {tag}: no result files =="); continue
        out = os.path.join(RESULTS_DIR, f"lexical_triviaqa_{tag}_summary.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"=== TriviaQA LEXICAL METRICS: model={tag} ===\nSizes present: {len(rows)}/4\n\n")
            for r in rows:
                f.write(f"size {r['size']} (n={r['n']}): cover_em={r['cover_em']:.4f}  token_recall={r['token_recall']:.4f}\n")
        print(f"== {tag} ==")
        for r in rows:
            print(f"  size {r['size']}: cover_em={r['cover_em']:.4f} token_recall={r['token_recall']:.4f}")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()

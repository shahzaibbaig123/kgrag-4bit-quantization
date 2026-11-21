"""
Paired F16-vs-Q4 correctness comparison on the new per-version SQUAD results.

For each size: pair F16 vs Q4 by sample id WITHIN each seed (same data file =>
same questions), pool the 5 seeds, then report wins/ties, pass-flips, mean diff,
and a paired significance test (Wilcoxon) + effect sizes (rank-biserial, Cohen's d_z).

  python score_compare_multi.py
  python score_compare_multi.py --sizes 200 500 1000 2000
"""
import os
import re
import argparse
import numpy as np
from statistics import mean, median

RES = "./SQUAD/results"
PASS = 4.0
TAGS = {"f16": "qwenf16", "q4": "qwenQ4"}

BLOCK = re.compile(
    r"#(\d+)\nQ:.*?\nGold:.*?\nAnswer:.*?\nScore:\s*([0-9.]+)\s*\|\s*Pass:\s*(True|False)",
    re.DOTALL,
)


def parse(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for m in BLOCK.finditer(open(path, encoding="utf-8").read()):
        out[int(m.group(1))] = {"score": float(m.group(2)), "pass": m.group(3) == "True"}
    return out


def collect_pairs(size):
    """Pool paired (f16, q4) score/pass over all 5 seeds. Returns list of dicts + n_seeds."""
    pairs, seeds = [], 0
    for v in range(5):
        a = parse(f"{RES}/eval_squad_{size}_v{v}_{TAGS['f16']}.txt")
        b = parse(f"{RES}/eval_squad_{size}_v{v}_{TAGS['q4']}.txt")
        common = set(a) & set(b)
        if not common:
            continue
        seeds += 1
        for i in sorted(common):
            pairs.append({"sf": a[i]["score"], "sq": b[i]["score"],
                          "pf": a[i]["pass"], "pq": b[i]["pass"]})
    return pairs, seeds


def stats(pairs):
    sf = np.array([p["sf"] for p in pairs])
    sq = np.array([p["sq"] for p in pairs])
    d = sf - sq
    n = len(d)
    res = {
        "n": n,
        "avg_f16": float(sf.mean()), "avg_q4": float(sq.mean()),
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "f16_wins": int((d > 0).sum()), "q4_wins": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "pass_f16": float(np.mean([p["pf"] for p in pairs])),
        "pass_q4": float(np.mean([p["pq"] for p in pairs])),
        "both_pass": sum(1 for p in pairs if p["pf"] and p["pq"]),
        "both_fail": sum(1 for p in pairs if not p["pf"] and not p["pq"]),
        "f16pass_q4fail": sum(1 for p in pairs if p["pf"] and not p["pq"]),
        "q4pass_f16fail": sum(1 for p in pairs if p["pq"] and not p["pf"]),
    }
    # paired significance + effect size on per-question score diffs
    nz = d[d != 0]
    res["n_nonzero"] = int(len(nz))
    if len(nz) == 0:
        res["wilcoxon_p"], res["rank_biserial"], res["cohen_dz"] = None, 0.0, 0.0
    else:
        from scipy.stats import wilcoxon, rankdata
        ranks = rankdata(np.abs(nz))
        tp, tm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        res["rank_biserial"] = float((tp - tm) / (tp + tm))
        res["cohen_dz"] = float(d.mean() / (d.std(ddof=1) + 1e-12))
        try:
            res["wilcoxon_p"] = float(wilcoxon(nz)[1])
        except Exception as e:
            res["wilcoxon_p"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000])
    args = ap.parse_args()

    out = f"{RES}/paired_f16_vs_q4_all.txt"
    csv = f"{RES}/paired_f16_vs_q4_all.csv"
    lines = ["Paired F16 vs Q4 correctness (pooled over 5 seeds per size)",
             f"Pass threshold: {PASS} | diff = F16 - Q4 | (+)=F16 better\n"]
    csv_rows = ["size,n,seeds,avg_f16,avg_q4,mean_diff,pass_f16,pass_q4,wilcoxon_p,rank_biserial,cohen_dz"]

    for size in args.sizes:
        pairs, seeds = collect_pairs(size)
        if not pairs:
            lines.append(f"size {size}: no paired results found\n"); continue
        s = stats(pairs)
        p_txt = "n/a (all identical)" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.4g}"
        sig = "" if s["wilcoxon_p"] is None else (
            "  -> NO significant difference" if s["wilcoxon_p"] > 0.05 else "  -> significant (p<0.05)")
        lines.append(
            f"=== size {size}  (n={s['n']} paired Qs over {seeds} seeds) ===\n"
            f"  avg score:  F16={s['avg_f16']:.4f}  Q4={s['avg_q4']:.4f}  mean_diff={s['mean_diff']:+.4f}  median_diff={s['median_diff']:+.1f}\n"
            f"  pass@{PASS}:  F16={s['pass_f16']:.4f}  Q4={s['pass_q4']:.4f}\n"
            f"  wins:  F16={s['f16_wins']}  Q4={s['q4_wins']}  ties={s['ties']}\n"
            f"  pass-flips:  both_pass={s['both_pass']}  both_fail={s['both_fail']}  "
            f"F16only={s['f16pass_q4fail']}  Q4only={s['q4pass_f16fail']}\n"
            f"  Wilcoxon p={p_txt}{sig}\n"
            f"  effect: rank-biserial={s['rank_biserial']:+.4f}  Cohen's d_z={s['cohen_dz']:+.4f}  (nonzero diffs={s['n_nonzero']})\n")
        csv_rows.append(f"{size},{s['n']},{seeds},{s['avg_f16']:.4f},{s['avg_q4']:.4f},"
                        f"{s['mean_diff']:.4f},{s['pass_f16']:.4f},{s['pass_q4']:.4f},"
                        f"{'' if s['wilcoxon_p'] is None else round(s['wilcoxon_p'],6)},"
                        f"{s['rank_biserial']:.4f},{s['cohen_dz']:.4f}")

    report = "\n".join(lines)
    open(out, "w").write(report)
    open(csv, "w").write("\n".join(csv_rows) + "\n")
    print(report)
    print(f"saved {out}\nsaved {csv}")


if __name__ == "__main__":
    main()

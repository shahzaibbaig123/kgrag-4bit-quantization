"""
Paired F16-vs-Q4 correctness comparison for TriviaQA (single seed), per size.

Pairs F16 vs Q4 by sample id within each size (same data file => same questions),
then reports wins/ties, mean diff, Wilcoxon p + effect sizes (rank-biserial,
Cohen's d_z). Reuses parse()/stats() from score_compare_multi.py.

Usage:  python score_compare_triviaqa.py
"""
import os
from score_compare_multi import parse, stats  # reuse read-only

RES = "./TRIVIAQA/results"
PASS = 4.0
TAGS = {"f16": "qwenf16", "q4": "qwenQ4"}
SIZES = [25, 50, 75, 100]


def collect_pairs(size):
    a = parse(f"{RES}/eval_triviaqa_{size}_{TAGS['f16']}.txt")
    b = parse(f"{RES}/eval_triviaqa_{size}_{TAGS['q4']}.txt")
    common = sorted(set(a) & set(b))
    return [{"sf": a[i]["score"], "sq": b[i]["score"], "pf": a[i]["pass"], "pq": b[i]["pass"]}
            for i in common]


def main():
    out = f"{RES}/paired_triviaqa.txt"
    csv = f"{RES}/paired_triviaqa.csv"
    lines = ["Paired F16 vs Q4 correctness, TriviaQA (single seed, per size)",
             f"Pass threshold: {PASS} | diff = F16 - Q4 | (+)=F16 better\n"]
    csv_rows = ["size,n,avg_f16,avg_q4,mean_diff,pass_f16,pass_q4,wilcoxon_p,rank_biserial,cohen_dz"]

    for size in SIZES:
        pairs = collect_pairs(size)
        if not pairs:
            lines.append(f"size {size}: no paired results found\n"); continue
        s = stats(pairs)
        p_txt = "n/a (all identical)" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.4g}"
        sig = "" if s["wilcoxon_p"] is None else (
            "  -> NO significant difference" if s["wilcoxon_p"] > 0.05 else "  -> significant (p<0.05)")
        lines.append(
            f"=== size {size}  (n={s['n']} paired Qs) ===\n"
            f"  avg score:  F16={s['avg_f16']:.4f}  Q4={s['avg_q4']:.4f}  mean_diff={s['mean_diff']:+.4f}  median_diff={s['median_diff']:+.1f}\n"
            f"  pass@{PASS}:  F16={s['pass_f16']:.4f}  Q4={s['pass_q4']:.4f}\n"
            f"  wins:  F16={s['f16_wins']}  Q4={s['q4_wins']}  ties={s['ties']}\n"
            f"  Wilcoxon p={p_txt}{sig}\n"
            f"  effect: rank-biserial={s['rank_biserial']:+.4f}  Cohen's d_z={s['cohen_dz']:+.4f}  (nonzero diffs={s['n_nonzero']})\n")
        csv_rows.append(f"{size},{s['n']},{s['avg_f16']:.4f},{s['avg_q4']:.4f},{s['mean_diff']:.4f},"
                        f"{s['pass_f16']:.4f},{s['pass_q4']:.4f},"
                        f"{'' if s['wilcoxon_p'] is None else round(s['wilcoxon_p'],6)},"
                        f"{s['rank_biserial']:.4f},{s['cohen_dz']:.4f}")

    report = "\n".join(lines)
    open(out, "w").write(report)
    open(csv, "w").write("\n".join(csv_rows) + "\n")
    print(report)
    print(f"saved {out}\nsaved {csv}")


if __name__ == "__main__":
    main()

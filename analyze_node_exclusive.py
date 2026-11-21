"""
Stage 2 (offline) of the model-exclusive-node QA probe.

Slices the ALREADY-COMPUTED full SQuAD eval results by node-exclusivity group
(no model re-run, no GPU). For each tagged question (from
node_exclusive_qa/subset_{size}_v{seed}.jsonl) it looks up the F16 and Q4
correctness score + answer from the existing eval .txt files, recomputes
Cover-EM / token-recall, and compares F16 vs Q4 WITHIN each group:

  f16_only : question about an entity only F16 has  -> is F16 > Q4 ?  (real loss?)
  q4_only  : question about an entity only Q4 has   -> is Q4  > F16?
  both     : about an entity both extracted
  control  : about an entity present in both graphs -> expect F16 ~= Q4 (anchor)

Non-destructive: this is a standalone script. It imports helpers read-only and
does NOT modify lexical_eval.py / score_compare_multi.py / node_exclusive_questions.py.

Usage:
  python analyze_node_exclusive.py                      # 200 500 1000 2000 3000
  python analyze_node_exclusive.py --sizes 200 500
"""
import os
import re
import json
import glob
import argparse
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse, read-only
from lexical_eval import normalize, cover_em, token_recall
from score_compare_multi import stats as paired_stats   # expects pairs of {sf,sq,pf,pq}

RESULTS_DIR = "SQUAD/results"
SUBSET_DIR = "node_exclusive_qa"
FIG_DIR = "presentation_figures"
PASS_THRESHOLD = 4.0
GROUPS = ["control", "both", "f16_only", "q4_only"]

# one Q/Answer/Score block in an eval .txt; ERROR blocks (no Score) won't match
_BLOCK = re.compile(
    r"#\d+\nQ:\s*(?P<q>.*?)\nGold:\s*(?P<gold>.*?)\nAnswer:\s*(?P<ans>.*?)\n"
    r"Score:\s*(?P<score>[\d.]+)\s*\|[^\n]*\n-{20,}",
    re.DOTALL,
)


def parse_eval_txt(path):
    """{question -> (score:float, answer:str)} from an eval result .txt (LOCAL, score-aware)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for m in _BLOCK.finditer(text):
        ans = m.group("ans").split("\n### References")[0].strip()   # drop refs block
        out[m.group("q").strip()] = (float(m.group("score")), ans)
    return out


def collect():
    """Walk every subset; join to F16/Q4 eval results. Returns rows + coverage stats."""
    rows = []                  # one per matched question, with all metrics
    seen = matched = 0
    for sub in sorted(glob.glob(os.path.join(SUBSET_DIR, "subset_*_v*.jsonl"))):
        base = os.path.basename(sub)
        size = int(base.split("_")[1]); seed = int(base.split("_v")[1].split(".")[0])
        f16 = parse_eval_txt(os.path.join(RESULTS_DIR, f"eval_squad_{size}_v{seed}_qwenf16.txt"))
        q4 = parse_eval_txt(os.path.join(RESULTS_DIR, f"eval_squad_{size}_v{seed}_qwenQ4.txt"))
        if not f16 or not q4:
            continue
        with open(sub, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line); seen += 1
                q = ex["question"].strip()
                if q not in f16 or q not in q4:
                    continue
                matched += 1
                golds = ex["answers"]
                sf, af = f16[q]; sq, aq = q4[q]
                rows.append({
                    "size": size, "seed": seed, "group": ex["group"],
                    "sf": sf, "sq": sq,
                    "pf": sf >= PASS_THRESHOLD, "pq": sq >= PASS_THRESHOLD,
                    "cf": cover_em(af, golds), "cq": cover_em(aq, golds),
                    "tf": token_recall(af, golds), "tq": token_recall(aq, golds),
                })
    return rows, seen, matched


def lex_pair(rows, fkey, qkey):
    """mean_f16, mean_q4, diff for a lexical metric over a list of rows."""
    if not rows:
        return 0.0, 0.0, 0.0
    mf = st.mean(r[fkey] for r in rows)
    mq = st.mean(r[qkey] for r in rows)
    return mf, mq, mf - mq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 3000])
    args = ap.parse_args()

    rows, seen, matched = collect()
    rows = [r for r in rows if r["size"] in args.sizes]
    by_group = {g: [r for r in rows if r["group"] == g] for g in GROUPS}

    summary = {"join_seen": seen, "join_matched": matched,
               "match_rate": round(matched / seen, 4) if seen else 0.0,
               "pass_threshold": PASS_THRESHOLD, "groups": {}}

    for g in GROUPS:
        grp = by_group[g]
        if not grp:
            continue
        s = paired_stats(grp)                       # correctness wins/ties/wilcoxon/d_z
        cf, cq, cd = lex_pair(grp, "cf", "cq")
        tf, tq, td = lex_pair(grp, "tf", "tq")
        summary["groups"][g] = {
            "n": s["n"],
            "corr_f16": round(s["avg_f16"], 4), "corr_q4": round(s["avg_q4"], 4),
            "corr_diff": round(s["mean_diff"], 4), "median_diff": s["median_diff"],
            "f16_wins": s["f16_wins"], "q4_wins": s["q4_wins"], "ties": s["ties"],
            "wilcoxon_p": s["wilcoxon_p"], "cohen_dz": round(s["cohen_dz"], 4),
            "rank_biserial": round(s["rank_biserial"], 4),
            "pass_f16": round(s["pass_f16"], 4), "pass_q4": round(s["pass_q4"], 4),
            "coverem_f16": round(cf, 4), "coverem_q4": round(cq, 4), "coverem_diff": round(cd, 4),
            "tokrec_f16": round(tf, 4), "tokrec_q4": round(tq, 4), "tokrec_diff": round(td, 4),
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- text report ---
    txt = os.path.join(RESULTS_DIR, "node_exclusive_summary.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("=== MODEL-EXCLUSIVE NODE QA PROBE (offline; sizes %s) ===\n" % args.sizes)
        f.write(f"Join: {matched}/{seen} subset questions matched an F16&Q4 eval score "
                f"(match rate {summary['match_rate']:.3f})\n")
        f.write("Hypotheses: f16_only -> F16>Q4 iff real loss; q4_only -> Q4>F16; "
                "control -> F16~=Q4 (anchor)\n\n")
        for g in GROUPS:
            d = summary["groups"].get(g)
            if not d:
                continue
            pstr = "n/a" if d["wilcoxon_p"] is None else f"{d['wilcoxon_p']:.4g}"
            f.write(f"[{g}]  n={d['n']}\n")
            f.write(f"  correctness : F16={d['corr_f16']:.3f}  Q4={d['corr_q4']:.3f}  "
                    f"diff(F16-Q4)={d['corr_diff']:+.3f}  median={d['median_diff']:+.1f}\n")
            f.write(f"  wins/ties   : F16={d['f16_wins']}  Q4={d['q4_wins']}  ties={d['ties']}\n")
            f.write(f"  significance: Wilcoxon p={pstr}  Cohen d_z={d['cohen_dz']:+.4f}  "
                    f"rank-biserial={d['rank_biserial']:+.4f}\n")
            f.write(f"  pass@4      : F16={d['pass_f16']:.3f}  Q4={d['pass_q4']:.3f}\n")
            f.write(f"  cover-EM    : F16={d['coverem_f16']:.3f}  Q4={d['coverem_q4']:.3f}  "
                    f"diff={d['coverem_diff']:+.3f}\n")
            f.write(f"  token-recall: F16={d['tokrec_f16']:.3f}  Q4={d['tokrec_q4']:.3f}  "
                    f"diff={d['tokrec_diff']:+.3f}\n\n")

    with open(os.path.join(RESULTS_DIR, "node_exclusive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv = ["group,n,corr_f16,corr_q4,corr_diff,wilcoxon_p,cohen_dz,pass_f16,pass_q4,"
           "coverem_f16,coverem_q4,tokrec_f16,tokrec_q4"]
    for g in GROUPS:
        d = summary["groups"].get(g)
        if not d:
            continue
        csv.append(f"{g},{d['n']},{d['corr_f16']},{d['corr_q4']},{d['corr_diff']},"
                   f"{d['wilcoxon_p']},{d['cohen_dz']},{d['pass_f16']},{d['pass_q4']},"
                   f"{d['coverem_f16']},{d['coverem_q4']},{d['tokrec_f16']},{d['tokrec_q4']}")
    with open(os.path.join(RESULTS_DIR, "node_exclusive_summary.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")

    # --- grouped bar chart: per-group F16 vs Q4 mean correctness ---
    glabels = [g for g in GROUPS if g in summary["groups"]]
    f16v = [summary["groups"][g]["corr_f16"] for g in glabels]
    q4v = [summary["groups"][g]["corr_q4"] for g in glabels]
    x = range(len(glabels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar([i - w/2 for i in x], f16v, w, label="F16", color="#1f77b4")
    b2 = ax.bar([i + w/2 for i in x], q4v, w, label="Q4", color="#d62728")
    ax.set_xticks(list(x)); ax.set_xticklabels(
        [f"{g}\n(n={summary['groups'][g]['n']})" for g in glabels])
    ax.set_ylabel("Mean correctness (1-5)")
    ax.set_ylim(min(f16v + q4v) - 0.25, max(f16v + q4v) + 0.25)
    ax.set_title("Answer correctness by node-exclusivity group (F16 vs Q4)")
    ax.legend()
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"node_exclusive_groups.{ext}"), dpi=150)

    print(f"join match rate: {summary['match_rate']:.3f}  ({matched}/{seen})")
    for g in glabels:
        d = summary["groups"][g]
        print(f"  {g:9s} n={d['n']:6d}  F16={d['corr_f16']:.3f} Q4={d['corr_q4']:.3f} "
              f"diff={d['corr_diff']:+.3f}  p={d['wilcoxon_p']}  d_z={d['cohen_dz']:+.3f}")
    print(f"saved: {txt} (+ .json/.csv) and {FIG_DIR}/node_exclusive_groups.png/pdf")


if __name__ == "__main__":
    main()

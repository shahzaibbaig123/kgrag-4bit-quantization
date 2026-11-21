"""
Generate the IEEE BigData paper's figures as column-width vector PDFs.

Outputs into paper/pics/ :
  fig_overlap.pdf     the exact-vs-semantic overlap cascade vs corpus size (RQ1)
  fig_decoupling.pdf  KG node Jaccard vs downstream correctness gap, per (size, seed) (RQ3)
  fig_nodeexcl.pdf    judge correctness by node-exclusivity group (the causal probe)

Everything derives from existing result files; no GPU and no re-evaluation:
  KG_results/kg_multi_comparison.json      per-size 'aggregate' + 'per_seed'
  KG_results/semantic_edge_overlap.json    semantic edge overlap
  SQUAD/results/eval_squad_{size}_v{v}_*.txt   per-seed correctness
  SQUAD/results/node_exclusive_summary.csv     the exclusivity probe

Geometry is IEEE single-column (3.4 in wide) with 8 pt tick / 9 pt axis labels.
Colors are categorical slots 1--2 of the validated reference palette, and every
series also carries a distinct linestyle/marker so the figures survive grayscale
printing.

Run from the repo root (the login node needs libpython on the path):
  LD_LIBRARY_PATH=/software/easybuild-INTEL_A40/software/Python/3.11.3-GCCcore-12.3.0/lib \
    ~/venvs/rag-py311/bin/python paper/make_paper_figures.py
"""
import os
import re
import csv
import json

# pin the PDF CreationDate so repeated runs produce byte-identical files and a
# stray re-run never shows up as a spurious diff (must precede the mpl import)
os.environ.setdefault("SOURCE_DATE_EPOCH", "1735689600")  # 2025-01-01T00:00:00Z

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "SQUAD/results")
KGJ = os.path.join(ROOT, "KG_results")
OUT = os.path.join(ROOT, "paper/pics")
os.makedirs(OUT, exist_ok=True)

SIZES = [200, 500, 1000, 2000, 3000]

# validated categorical slots 1 and 2 (light surface) + the blue sequential ramp
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#104281"]  # ordinal-safe steps
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

COL_W = 3.4  # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
})

kg = json.load(open(f"{KGJ}/kg_multi_comparison.json"))
semedge = json.load(open(f"{KGJ}/semantic_edge_overlap.json"))


def agg(size, key):
    a = kg[str(size)]["aggregate"][key]
    return a["mean"], a["std"]


def per_seed(size, key):
    return np.array([r.get(key, np.nan) for r in kg[str(size)]["per_seed"]], dtype=float)


def seed_corr(size, tag):
    """per-seed average judge correctness, from the per-seed summary .txt files."""
    out = []
    for v in range(5):
        p = f"{RES}/eval_squad_{size}_v{v}_{tag}.txt"
        if not os.path.exists(p):
            out.append(np.nan)
            continue
        m = re.search(r"Average correctness:\s*([0-9.]+)", open(p).read())
        out.append(float(m.group(1)) if m else np.nan)
    return np.array(out)


def tidy(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    p = f"{OUT}/{name}.pdf"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


# --------------------------------------------------------------------------
# fig_overlap -- the exact/semantic cascade
#   hue encodes the graph element (nodes / edges); linestyle encodes the
#   matching criterion (exact / semantic). Two hues, four series, no cycling.
# --------------------------------------------------------------------------
def fig_overlap():
    x = np.arange(len(SIZES))
    series = [
        ("Nodes, semantic", [agg(s, "sem_recall_f16_in_q4") for s in SIZES], BLUE, "-", "o"),
        ("Edges, semantic", [tuple(semedge[str(s)]["sem_edge_recall"]) for s in SIZES], ORANGE, "-", "s"),
        ("Nodes, exact", [agg(s, "node_jaccard") for s in SIZES], BLUE, "--", "o"),
        ("Edges, exact", [tuple(semedge[str(s)]["edge_jaccard"]) for s in SIZES], ORANGE, "--", "s"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for label, vals, c, ls, mk in series:
        m = np.array([v[0] for v in vals])
        e = np.array([v[1] for v in vals])
        ax.errorbar(x, m, yerr=e, color=c, linestyle=ls, marker=mk, markersize=3.5,
                    linewidth=1.4, capsize=2, elinewidth=0.8,
                    markerfacecolor="white" if ls == "--" else c,
                    markeredgewidth=1.0, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("Corpus size (indexed paragraphs)")
    ax.set_ylabel("F16$\\leftrightarrow$Q4 overlap")
    ax.set_ylim(0.15, 0.95)
    # the band between exact-node (~0.62) and semantic-node (~0.79) overlap is the
    # only region no series crosses; park the legend there so nothing is occluded
    ax.legend(ncol=2, loc="center", bbox_to_anchor=(0.5, 0.66), handlelength=2.0,
              columnspacing=0.9, labelspacing=0.2, borderaxespad=0.0, fontsize=7)
    tidy(ax)
    save(fig, "fig_overlap")


# --------------------------------------------------------------------------
# fig_decoupling -- structural drift vs downstream gap
# --------------------------------------------------------------------------
def fig_decoupling():
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    markers = ["o", "s", "^", "D", "v"]
    for i, s in enumerate(SIZES):
        jac = per_seed(s, "node_jaccard")
        gap = seed_corr(s, "qwenf16") - seed_corr(s, "qwenQ4")
        ax.scatter(jac, gap, s=26, c=SEQ[i], marker=markers[i], linewidths=0.6,
                   edgecolors="white", label=f"{s}", zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":", zorder=2)
    ax.set_xlabel("Exact KG node Jaccard")
    ax.set_ylabel("Correctness gap (F16$-$Q4)")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(title="Corpus size", ncol=2, loc="upper left", handletextpad=0.2,
              labelspacing=0.25, columnspacing=0.8, borderaxespad=0.2,
              title_fontsize=7.5)
    tidy(ax)
    save(fig, "fig_decoupling")


# --------------------------------------------------------------------------
# fig_nodeexcl -- the model-exclusive-entity probe
# --------------------------------------------------------------------------
GROUPS = [
    ("q4_only", "Q4-only entity"),
    ("f16_only", "F16-only entity"),
    ("both", "One exclusive from each"),
    ("control", "Shared entities only"),
]
DIV_POS, DIV_NEG = "#2a78d6", "#d03b3b"  # diverging poles: F16 ahead / Q4 ahead


def fig_nodeexcl():
    """The quantity of interest is the signed gap, which has a true zero, so this is
    a diverging dot plot rather than bars on a truncated correctness axis."""
    recs = {r["group"]: r for r in csv.DictReader(open(f"{RES}/node_exclusive_summary.csv"))}
    y = np.arange(len(GROUPS))
    diffs = [float(recs[g]["corr_diff"]) for g, _ in GROUPS]
    ns = [int(recs[g]["n"]) for g, _ in GROUPS]

    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    for i, d in enumerate(diffs):
        c = DIV_POS if d > 0 else DIV_NEG
        ax.plot([0, d], [i, i], color=c, linewidth=1.6, zorder=3, solid_capstyle="round")
        ax.plot([d], [i], marker="o", markersize=6.5, color=c, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        off = 9 if d > 0 else -9
        ax.annotate(f"{d:+.3f}", xy=(d, i), xytext=(off, 0), textcoords="offset points",
                    va="center", ha="left" if d > 0 else "right", fontsize=7, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n($n$={n:,})" for (_, lab), n in zip(GROUPS, ns)])
    ax.set_xlabel("Correctness gap (F16 $-$ Q4)")
    ax.set_xlim(-0.24, 0.24)
    ax.set_ylim(-0.5, len(GROUPS) - 0.15)
    ax.annotate("$\\leftarrow$ Q4 ahead", xy=(-0.23, len(GROUPS) - 0.42), fontsize=7,
                color=DIV_NEG, ha="left", va="center")
    ax.annotate("F16 ahead $\\rightarrow$", xy=(0.23, len(GROUPS) - 0.42), fontsize=7,
                color=DIV_POS, ha="right", va="center")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_nodeexcl")


if __name__ == "__main__":
    fig_overlap()
    fig_decoupling()
    fig_nodeexcl()

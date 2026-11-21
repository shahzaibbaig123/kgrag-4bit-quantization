"""
KG structure comparison F16 vs Q4 for the TriviaQA indexes (single seed).

Reuses the metric functions from eval_KG_multi.py; only the paths differ
(TRIVIAQA/{model}/{size}/ with no seed suffix). Per size 25/50/75/100.
Offline (reads graphml; the embedder for --semantic runs on CPU/GPU).

Output: KG_results/kg_triviaqa_comparison.json + kg_triviaqa_summary.txt

Usage:
  python eval_KG_triviaqa.py --semantic         # all sizes, with semantic overlap
  python eval_KG_triviaqa.py --sizes 25 50
"""
import os, json, argparse
import numpy as np
import networkx as nx

from eval_KG_multi import basic, degree, connectivity, overlaps, topo, semantic_overlap

OUT = "KG_results"
os.makedirs(OUT, exist_ok=True)


def gpath(model, size):
    return f"TRIVIAQA/{model}/{size}/graph_chunk_entity_relation.graphml"


def compare_size(size, sem_model=None, sem_threshold=0.85):
    Gf = nx.read_graphml(gpath("f16", size))
    Gq = nx.read_graphml(gpath("q4", size))
    bf, bq = basic(Gf), basic(Gq)
    res = {
        "size": size,
        "nodes_f16": bf["nodes"], "nodes_q4": bq["nodes"],
        "edges_f16": bf["edges"], "edges_q4": bq["edges"],
        "reltypes_f16": bf["relation_types"], "reltypes_q4": bq["relation_types"],
        "mean_degree_f16": degree(Gf)["mean_degree"], "mean_degree_q4": degree(Gq)["mean_degree"],
        "giant_pct_f16": connectivity(Gf)["giant_pct"], "giant_pct_q4": connectivity(Gq)["giant_pct"],
        **overlaps(Gf, Gq),
        **topo(Gf, Gq),
    }
    if sem_model is not None:
        res.update(semantic_overlap(Gf, Gq, sem_model, sem_threshold))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 75, 100])
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--sem-threshold", type=float, default=0.85)
    args = ap.parse_args()

    sem_model = None
    if args.semantic:
        from sentence_transformers import SentenceTransformer
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")

    all_results, lines = {}, []
    for size in args.sizes:
        if not (os.path.exists(gpath("f16", size)) and os.path.exists(gpath("q4", size))):
            print(f"size {size}: graphs missing, skip"); continue
        print(f"  comparing TriviaQA size {size} ...")
        r = compare_size(size, sem_model=sem_model, sem_threshold=args.sem_threshold)
        all_results[str(size)] = r
        sem = (f"  sem_recall={r['sem_recall_f16_in_q4']:.3f} sem_prec={r['sem_precision_q4_in_f16']:.3f}\n"
               if "sem_recall_f16_in_q4" in r else "")
        lines.append(
            f"size {size}:\n"
            f"  nodes  F16={r['nodes_f16']}  Q4={r['nodes_q4']}  (Q4/F16={r['nodes_q4']/r['nodes_f16']:.3f})\n"
            f"  edges  F16={r['edges_f16']}  Q4={r['edges_q4']}  (Q4/F16={r['edges_q4']/r['edges_f16']:.3f})\n"
            f"  node_jaccard={r['node_jaccard']:.3f}  edge_jaccard={r['edge_jaccard']:.3f}  hub_jaccard@50={r['hub_jaccard_top50']:.3f}\n"
            f"  mean_degree F16={r['mean_degree_f16']:.2f} Q4={r['mean_degree_q4']:.2f}  giant% F16={r['giant_pct_f16']:.3f} Q4={r['giant_pct_q4']:.3f}\n"
            + sem)

    with open(os.path.join(OUT, "kg_triviaqa_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT, "kg_triviaqa_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"saved {OUT}/kg_triviaqa_comparison.json + kg_triviaqa_summary.txt")


if __name__ == "__main__":
    main()

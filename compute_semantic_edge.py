"""
Compute semantic EDGE overlap for the SQuAD F16-vs-Q4 graph pairs (all sizes/seeds).
Non-destructive: prints a per-size summary and writes KG_results/semantic_edge_overlap.json.
Reuses semantic_edge_overlap() from eval_KG_multi.py; also reports the corrected exact
edge Jaccard (now an ordered (source,target) entity-pair overlap) and semantic node overlap
for context.

Run (login node):
  LD_LIBRARY_PATH=... ~/venvs/rag-a100/bin/python compute_semantic_edge.py
"""
import json, statistics
import networkx as nx
from sentence_transformers import SentenceTransformer
from eval_KG_multi import gpath, overlaps, semantic_overlap, semantic_edge_overlap

SIZES = [200, 500, 1000, 2000, 3000]
THRESH = 0.85

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    out = {}
    for size in SIZES:
        rows = []
        for v in range(5):
            Gf = nx.read_graphml(gpath("f16", size, v))
            Gq = nx.read_graphml(gpath("q4", size, v))
            ov = overlaps(Gf, Gq)
            sn = semantic_overlap(Gf, Gq, model, THRESH)
            se = semantic_edge_overlap(Gf, Gq, model, THRESH)
            rows.append({
                "v": v,
                "edge_jaccard": ov["edge_jaccard"],
                "sem_node_recall": sn["sem_recall_f16_in_q4"],
                "sem_node_prec": sn["sem_precision_q4_in_f16"],
                "sem_edge_recall": se["sem_edge_recall_f16_in_q4"],
                "sem_edge_prec": se["sem_edge_precision_q4_in_f16"],
            })
            print(f"  size {size} v{v}: exact_edge_J={ov['edge_jaccard']:.3f} "
                  f"sem_edge_rec={se['sem_edge_recall_f16_in_q4']:.3f} "
                  f"sem_edge_prec={se['sem_edge_precision_q4_in_f16']:.3f}", flush=True)
        def ms(key):
            xs = [r[key] for r in rows]
            return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)
        out[str(size)] = {k: ms(k) for k in
                          ["edge_jaccard", "sem_node_recall", "sem_node_prec",
                           "sem_edge_recall", "sem_edge_prec"]}
    json.dump(out, open("KG_results/semantic_edge_overlap.json", "w"), indent=2)
    print("\n=== SUMMARY (5-seed mean +/- s.d.) ===")
    print(f"{'size':>5} | {'exact edge J':>13} | {'sem edge recall':>16} | {'sem edge prec':>14} | {'sem node rec':>13}")
    for size in SIZES:
        d = out[str(size)]
        f = lambda k: f"{d[k][0]:.3f}+/-{d[k][1]:.3f}"
        print(f"{size:>5} | {f('edge_jaccard'):>13} | {f('sem_edge_recall'):>16} | {f('sem_edge_prec'):>14} | {f('sem_node_recall'):>13}")
    print("\nwrote KG_results/semantic_edge_overlap.json")

if __name__ == "__main__":
    main()

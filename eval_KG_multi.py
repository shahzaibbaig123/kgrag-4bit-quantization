"""
KG structure comparison F16 vs Q4 for the multi-seed SQuAD indexes.

Compares F16-vN vs Q4-vN (same seed => same documents => fair), per size,
then aggregates across the 5 seeds (mean +/- std). Offline (reads graphml; no GPU).

Output: KG_results/kg_multi_comparison.json + KG_results/kg_multi_summary.txt
        + KG_results/degree_<size>_q4_vs_f16.png

Usage:
  python eval_KG_multi.py                  # sizes 200 500 1000 2000
  python eval_KG_multi.py --sizes 200 500  # subset
"""
import os, json, argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "KG_results"
os.makedirs(OUT, exist_ok=True)


def gpath(model, size, v):
    return f"SQUAD/{model}/{size}/v{v}/graph_chunk_entity_relation.graphml"

def basic(G):
    rel = set()
    for _, _, d in G.edges(data=True):
        rel.add(d.get("relation", d.get("label", "")))
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "relation_types": len(rel)}

def degree(G):
    d = np.array([x for _, x in G.degree()]) if G.number_of_nodes() else np.array([0])
    return {"mean_degree": float(d.mean()), "median_degree": float(np.median(d)),
            "p95_degree": float(np.percentile(d, 95)), "max_degree": int(d.max())}

def connectivity(G):
    comps = sorted((len(c) for c in nx.connected_components(G.to_undirected())), reverse=True)
    n = G.number_of_nodes() or 1
    return {"num_components": len(comps), "giant_pct": comps[0] / n if comps else 0.0}

def top_hubs(G, k=50):
    return {n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:k]}

def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def overlaps(Gf, Gq):
    nf, nq = set(Gf.nodes()), set(Gq.nodes())
    # An edge is an ordered entity pair (source, target). Exact edge overlap therefore
    # requires BOTH endpoint labels to coincide; the relation content is NOT part of this
    # exact test (exact relation-string matching is far too strict, ~0.03) and is instead
    # compared semantically in semantic_edge_overlap().
    ef = set(Gf.edges())
    eq = set(Gq.edges())
    return {
        "node_jaccard": jaccard(nf, nq),
        "node_recall_f16_in_q4": (len(nf & nq) / len(nf)) if nf else 0.0,
        "edge_jaccard": jaccard(ef, eq),
        "hub_jaccard_top50": jaccard(top_hubs(Gf), top_hubs(Gq)),
    }

def topo(Gf, Gq):
    try:
        a_f = nx.degree_assortativity_coefficient(Gf)
        a_q = nx.degree_assortativity_coefficient(Gq)
        assort = abs(a_f - a_q)
    except Exception:
        assort = None
    return {"clustering_diff": abs(nx.average_clustering(Gf.to_undirected())
                                   - nx.average_clustering(Gq.to_undirected())),
            "assortativity_diff": assort}

def semantic_overlap(Gf, Gq, model, threshold=0.85, batch=1024):
    """Batched (memory-safe) semantic node overlap via MiniLM cosine, max-match per node."""
    nf, nq = list(Gf.nodes()), list(Gq.nodes())
    if not nf or not nq:
        return {"sem_recall_f16_in_q4": 0.0, "sem_precision_q4_in_f16": 0.0, "sem_threshold": threshold}
    ef = model.encode(nf, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    eq = model.encode(nq, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    def frac_matched(A, B):
        cnt = 0
        for i in range(0, A.shape[0], batch):
            sims = A[i:i + batch] @ B.T          # cosine (already normalized)
            cnt += int((sims.max(dim=1).values >= threshold).sum().item())
        return cnt

    return {"sem_recall_f16_in_q4": frac_matched(ef, eq) / len(nf),   # F16 nodes with a Q4 match
            "sem_precision_q4_in_f16": frac_matched(eq, ef) / len(nq), # Q4 nodes with an F16 match
            "sem_threshold": threshold}


def _edge_string(u, v, d):
    """Linearize an edge to a short relational statement: 'source [relation] target'.
    Uses the extracted relation keywords (falls back to the description)."""
    rel = (d.get("keywords") or d.get("description") or "").strip()
    return f"{u} {rel} {v}".strip()

def semantic_edge_overlap(Gf, Gq, model, threshold=0.85, batch=1024):
    """Semantic edge overlap: two edges match if their linearized relational statements
    ('source relation target') are cosine-similar >= threshold. Unlike exact edge Jaccard,
    this DOES compare the relation content, not just which entity pairs are connected."""
    ef = [_edge_string(u, v, d) for u, v, d in Gf.edges(data=True)]
    eq = [_edge_string(u, v, d) for u, v, d in Gq.edges(data=True)]
    if not ef or not eq:
        return {"sem_edge_recall_f16_in_q4": 0.0, "sem_edge_precision_q4_in_f16": 0.0,
                "sem_edge_threshold": threshold}
    Ef = model.encode(ef, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    Eq = model.encode(eq, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    def frac_matched(A, B):
        cnt = 0
        for i in range(0, A.shape[0], batch):
            sims = A[i:i + batch] @ B.T
            cnt += int((sims.max(dim=1).values >= threshold).sum().item())
        return cnt

    return {"sem_edge_recall_f16_in_q4": frac_matched(Ef, Eq) / len(ef),
            "sem_edge_precision_q4_in_f16": frac_matched(Eq, Ef) / len(eq),
            "sem_edge_threshold": threshold}


def compare_seed(size, v, sem_model=None, sem_threshold=0.85):
    Gf = nx.read_graphml(gpath("f16", size, v))
    Gq = nx.read_graphml(gpath("q4", size, v))
    bf, bq = basic(Gf), basic(Gq)
    res = {
        "version": v,
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
        res.update(semantic_edge_overlap(Gf, Gq, sem_model, sem_threshold))
    res["_Gf_deg"] = [d for _, d in Gf.degree()]
    res["_Gq_deg"] = [d for _, d in Gq.degree()]
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000])
    ap.add_argument("--semantic", action="store_true", help="also compute semantic node overlap (MiniLM)")
    ap.add_argument("--sem-threshold", type=float, default=0.85)
    args = ap.parse_args()

    sem_model = None
    if args.semantic:
        from sentence_transformers import SentenceTransformer
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")

    AGG_KEYS = ["nodes_f16", "nodes_q4", "edges_f16", "edges_q4",
                "mean_degree_f16", "mean_degree_q4", "giant_pct_f16", "giant_pct_q4",
                "node_jaccard", "edge_jaccard", "hub_jaccard_top50",
                "node_recall_f16_in_q4", "clustering_diff", "assortativity_diff"]
    if args.semantic:
        AGG_KEYS += ["sem_recall_f16_in_q4", "sem_precision_q4_in_f16"]
    all_results, summary_lines = {}, []

    for size in args.sizes:
        if not os.path.exists(gpath("f16", size, 0)) or not os.path.exists(gpath("q4", size, 0)):
            print(f"size {size}: graphs missing, skip"); continue
        per_seed, degs_f, degs_q = [], [], []
        for v in range(5):
            if not (os.path.exists(gpath("f16", size, v)) and os.path.exists(gpath("q4", size, v))):
                continue
            print(f"  comparing size {size} v{v} ...")
            r = compare_seed(size, v, sem_model=sem_model, sem_threshold=args.sem_threshold)
            degs_f += r.pop("_Gf_deg"); degs_q += r.pop("_Gq_deg")
            per_seed.append(r)
        if not per_seed:
            continue
        agg = {}
        for k in AGG_KEYS:
            vals = [r[k] for r in per_seed if r.get(k) is not None]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
        all_results[str(size)] = {"seeds": len(per_seed), "per_seed": per_seed, "aggregate": agg}

        # degree plot (pooled across seeds)
        plt.figure(figsize=(8, 5))
        plt.hist(degs_f, bins=50, alpha=0.6, log=True, label=f"f16_{size}")
        plt.hist(degs_q, bins=50, alpha=0.6, log=True, label=f"q4_{size}")
        plt.legend(); plt.xlabel("Degree"); plt.ylabel("Count (log)")
        plt.title(f"Degree dist: Q4 vs F16 (size {size}, pooled seeds)")
        pp = os.path.join(OUT, f"degree_{size}_q4_vs_f16.png")
        plt.savefig(pp, dpi=200, bbox_inches="tight"); plt.close()

        nf, nq = agg["nodes_f16"]["mean"], agg["nodes_q4"]["mean"]
        ef, eq = agg["edges_f16"]["mean"], agg["edges_q4"]["mean"]
        summary_lines.append(
            f"size {size} ({len(per_seed)} seeds):\n"
            f"  nodes  F16={nf:.0f}  Q4={nq:.0f}  (Q4/F16={nq/nf:.3f})\n"
            f"  edges  F16={ef:.0f}  Q4={eq:.0f}  (Q4/F16={eq/ef:.3f})\n"
            f"  node_jaccard={agg['node_jaccard']['mean']:.3f}  edge_jaccard={agg['edge_jaccard']['mean']:.3f}  hub_jaccard@50={agg['hub_jaccard_top50']['mean']:.3f}\n"
            f"  mean_degree F16={agg['mean_degree_f16']['mean']:.2f} Q4={agg['mean_degree_q4']['mean']:.2f}  giant%% F16={agg['giant_pct_f16']['mean']:.3f} Q4={agg['giant_pct_q4']['mean']:.3f}\n"
        )

    with open(os.path.join(OUT, "kg_multi_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT, "kg_multi_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\n" + "\n".join(summary_lines))
    print(f"saved {OUT}/kg_multi_comparison.json + kg_multi_summary.txt + degree plots")


if __name__ == "__main__":
    main()

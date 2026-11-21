"""
Publication-quality static KG figures (PNG + PDF) for the thesis.

The full KG is an unreadable hairball, so we draw a focused, connected subgraph
(ego-network of a high-degree hub) and color nodes by SEMANTIC overlap with the
other model's KG:
  blue = entity reproduced in the other model (>=threshold semantic match)
  red  = only in F16   /   green = only in Q4

Two figures per run (F16 perspective + Q4 perspective).
  python viz_paper.py --size 200 --version 0
  python viz_paper.py --size 200 --version 0 --max-nodes 60 --radius 1
"""
import os
import argparse
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_graph(path):
    return nx.read_graphml(path)


def semantic_matched_set(G, other_G, model, threshold=0.85, batch=1024):
    g, o = list(G.nodes()), list(other_G.nodes())
    if not g or not o:
        return set()
    eg = model.encode(g, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    eo = model.encode(o, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    matched = set()
    for i in range(0, eg.shape[0], batch):
        sims = eg[i:i + batch] @ eo.T
        for j, h in enumerate((sims.max(dim=1).values >= threshold).tolist()):
            if h:
                matched.add(g[i + j])
    return matched


def pick_subgraph(G, radius, max_nodes):
    """Ego-network of the highest-degree node; trimmed to top-degree neighbors if too big."""
    hub = max(G.degree, key=lambda x: x[1])[0]
    ego = nx.ego_graph(G.to_undirected(), hub, radius=radius)
    if ego.number_of_nodes() > max_nodes:
        keep = [hub] + [n for n, _ in sorted(ego.degree, key=lambda x: x[1], reverse=True)
                        if n != hub][:max_nodes - 1]
        ego = ego.subgraph(keep).copy()
    return ego, hub


def draw(G_self, other_G, matched, out_base, title, self_color, only_label,
         radius, max_nodes, n_hub_labels=12):
    SG, hub = pick_subgraph(G_self, radius, max_nodes)
    deg = dict(G_self.degree())
    colors = ["#3a7bd5" if n in matched else self_color for n in SG.nodes()]   # blue if shared
    sizes = [120 + 60 * deg.get(n, 1) for n in SG.nodes()]

    pos = nx.spring_layout(SG, seed=42, k=1.2 / (SG.number_of_nodes() ** 0.5))
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(SG, pos, edge_color="#bbbbbb", width=0.6, alpha=0.6)
    nx.draw_networkx_nodes(SG, pos, node_color=colors, node_size=sizes,
                           edgecolors="#333333", linewidths=0.4)
    # label only the top hubs in the subgraph to avoid clutter
    top = dict(sorted(SG.degree, key=lambda x: x[1], reverse=True)[:n_hub_labels])
    nx.draw_networkx_labels(SG, pos, labels={n: n for n in top}, font_size=7)

    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a7bd5",
                     markersize=10, label="shared (semantic match in other model)"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=self_color,
                     markersize=10, label=only_label)]
    plt.legend(handles=legend, loc="upper right", fontsize=10, framealpha=0.95)
    plt.title(title, fontsize=13)
    plt.axis("off"); plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_base}.png / .pdf  (subgraph: {SG.number_of_nodes()} nodes, hub='{hub}')")


def draw_macro(G, matched, out_base, title, only_color, only_label):
    """Whole-graph overview: tiny nodes, no labels, colored by semantic overlap."""
    deg = dict(G.degree())
    n = G.number_of_nodes()
    shared = len(matched)
    colors = ["#3a7bd5" if x in matched else only_color for x in G.nodes()]
    sizes = [6 + 3 * deg.get(x, 1) for x in G.nodes()]
    print(f"  macro layout for {n} nodes ...")
    pos = nx.spring_layout(G, seed=42, iterations=40, k=2.5 / (n ** 0.5))
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, edge_color="#cccccc", width=0.25, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, linewidths=0)
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a7bd5",
                     markersize=10, label=f"shared (semantic): {shared}/{n} = {shared/n:.0%}"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=only_color,
                     markersize=10, label=f"{only_label}: {n-shared}")]
    plt.legend(handles=legend, loc="upper right", fontsize=10, framealpha=0.95)
    plt.title(title, fontsize=13)
    plt.axis("off"); plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_base}.png / .pdf  (macro: {n} nodes, {shared/n:.0%} shared)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=80)
    ap.add_argument("--sem-threshold", type=float, default=0.85)
    args = ap.parse_args()

    f16p = f"SQUAD/f16/{args.size}/v{args.version}/graph_chunk_entity_relation.graphml"
    q4p = f"SQUAD/q4/{args.size}/v{args.version}/graph_chunk_entity_relation.graphml"
    for p in (f16p, q4p):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    out_dir = f"graphs/paper/{args.size}/v{args.version}"
    os.makedirs(out_dir, exist_ok=True)

    Gf, Gq = load_graph(f16p), load_graph(q4p)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("computing semantic matches ...")
    m_f16 = semantic_matched_set(Gf, Gq, model, args.sem_threshold)
    m_q4 = semantic_matched_set(Gq, Gf, model, args.sem_threshold)

    tag = f"SQuAD size {args.size}, seed v{args.version}"
    # --- MACRO: whole-graph overview (scale + overall overlap) ---
    draw_macro(Gf, m_f16, f"{out_dir}/kg_f16_macro",
               f"F16 KG overview ({tag})", only_color="#d9534f", only_label="only in F16")
    draw_macro(Gq, m_q4, f"{out_dir}/kg_q4_macro",
               f"Q4 KG overview ({tag})", only_color="#5cb85c", only_label="only in Q4")
    # --- ZOOM: ego-network of top hub (entity-level detail) ---
    draw(Gf, Gq, m_f16, f"{out_dir}/kg_f16_zoom",
         f"F16 KG — hub detail ({tag})\nblue = entity also in Q4 (semantic), red = F16-only",
         self_color="#d9534f", only_label="only in F16",
         radius=args.radius, max_nodes=args.max_nodes)
    draw(Gq, Gf, m_q4, f"{out_dir}/kg_q4_zoom",
         f"Q4 KG — hub detail ({tag})\nblue = entity also in F16 (semantic), green = Q4-only",
         self_color="#5cb85c", only_label="only in Q4",
         radius=args.radius, max_nodes=args.max_nodes)
    print(f"done -> {out_dir}/")


if __name__ == "__main__":
    main()

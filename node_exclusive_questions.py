"""
Stage 1 of the model-exclusive-node QA probe (offline, CPU).

For each (size, seed) it:
  1. loads the F16 and Q4 knowledge graphs,
  2. finds SEMANTICALLY model-exclusive entities (in F16 but not Q4 and vice
     versa) using MiniLM cosine matching (threshold 0.85) -- same logic as viz.py,
  3. tags every frozen SQuAD question by which kind of entity it is "about":
       - f16_only : mentions >=1 F16-only entity and no Q4-only entity
       - q4_only  : mentions >=1 Q4-only entity and no F16-only entity
       - both     : mentions both kinds
       - control  : mentions only entities present in BOTH graphs (no-effect anchor)
     "about" = the (normalized, >=min-len) entity label appears as a whole-word
     phrase in the question text OR in a gold answer.
  4. writes node_exclusive_qa/subset_{size}_v{seed}.jsonl + manifest.json

No LLM servers needed; run in the rag-a100 venv.

Usage:
  python node_exclusive_questions.py                       # 200/500/1000/2000 x v0-4
  python node_exclusive_questions.py --sizes 200 --seeds 0
"""
import os
import json
import argparse

import networkx as nx
from sentence_transformers import SentenceTransformer

# reuse existing helpers (single source of truth)
from viz import semantic_matched_set            # MiniLM cosine matched set
from lexical_eval import normalize              # SQuAD-style normalization

GRAPH_FILE = "graph_chunk_entity_relation.graphml"


def graph_path(model, size, seed):
    return f"SQUAD/{model}/{size}/v{seed}/{GRAPH_FILE}"


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def node_source_docs(node, entity_chunks, text_chunks):
    """First source doc id(s) for an entity, via entity->chunk->doc maps."""
    info = entity_chunks.get(node)
    if not info:
        return []
    docs = []
    for cid in info.get("chunk_ids", []):
        ch = text_chunks.get(cid)
        if ch and ch.get("full_doc_id"):
            docs.append(ch["full_doc_id"])
    return sorted(set(docs))


def norm_label(lbl, min_len):
    """Normalize an entity label to a whole-word phrase; '' if unusable.

    Rejects labels that are too short or purely numeric/date-like (no alphabetic
    char) -- a model-exclusive number/date is extraction noise, not lost knowledge,
    and both models can still answer it from the shared chunk.
    """
    n = normalize(lbl)
    if len(n) < min_len:
        return ""
    if not any(ch.isalpha() for ch in n):
        return ""
    return n


def build_phrase_index(labels, min_len):
    """labels(set[str]) -> (set of normalized phrases, max token count, phrase->orig)."""
    phrases = {}
    maxlen = 1
    for lbl in labels:
        n = norm_label(lbl, min_len)
        if not n:
            continue
        phrases.setdefault(n, lbl)            # keep one original spelling
        maxlen = max(maxlen, len(n.split()))
    return set(phrases), maxlen, phrases


def phrase_hits(text, phrase_set, maxlen):
    """Return the set of normalized phrases from phrase_set that occur as whole-word n-grams in text."""
    toks = normalize(text).split()
    if not toks:
        return set()
    hits = set()
    n = len(toks)
    for size in range(1, min(maxlen, n) + 1):
        for i in range(n - size + 1):
            ng = " ".join(toks[i:i + size])
            if ng in phrase_set:
                hits.add(ng)
    return hits


def process_seed(size, seed, model, args):
    gpf, gpq = graph_path("f16", size, seed), graph_path("q4", size, seed)
    if not (os.path.exists(gpf) and os.path.exists(gpq)):
        print(f"  [skip] missing graph(s) for size={size} v{seed}")
        return None

    Gf, Gq = nx.read_graphml(gpf), nx.read_graphml(gpq)
    nodes_f, nodes_q = set(Gf.nodes()), set(Gq.nodes())

    matched_f = semantic_matched_set(Gf, Gq, model, threshold=args.threshold)  # F16 nodes w/ Q4 twin
    matched_q = semantic_matched_set(Gq, Gf, model, threshold=args.threshold)  # Q4 nodes w/ F16 twin
    f16_only = nodes_f - matched_f
    q4_only = nodes_q - matched_q
    both = matched_f | matched_q                                              # present in both graphs

    mlen = args.min_label_len
    f_set, f_max, f_map = build_phrase_index(f16_only, mlen)
    q_set, q_max, q_map = build_phrase_index(q4_only, mlen)
    b_set, b_max, b_map = build_phrase_index(both, mlen)

    # source-doc provenance (owning model's maps)
    ec_f = load_json(f"SQUAD/f16/{size}/v{seed}/kv_store_entity_chunks.json")
    tc_f = load_json(f"SQUAD/f16/{size}/v{seed}/kv_store_text_chunks.json")
    ec_q = load_json(f"SQUAD/q4/{size}/v{seed}/kv_store_entity_chunks.json")
    tc_q = load_json(f"SQUAD/q4/{size}/v{seed}/kv_store_text_chunks.json")

    data_path = f"data/squad_train_{size}_v{seed}.jsonl"
    out_path = os.path.join(args.out_dir, f"subset_{size}_v{seed}.jsonl")
    counts = {"f16_only": 0, "q4_only": 0, "both": 0, "control": 0}

    with open(data_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            q = (ex.get("question") or "").strip()
            golds = [g for g in ex.get("answers", {}).get("text", []) if g.strip()]
            if not q or not golds:
                continue
            ans_text = " ".join(golds)

            f_in_q = phrase_hits(q, f_set, f_max)
            f_in_a = phrase_hits(ans_text, f_set, f_max)
            q_in_q = phrase_hits(q, q_set, q_max)
            q_in_a = phrase_hits(ans_text, q_set, q_max)
            has_f = bool(f_in_q or f_in_a)
            has_q = bool(q_in_q or q_in_a)

            if has_f and has_q:
                group = "both"
            elif has_f:
                group = "f16_only"
            elif has_q:
                group = "q4_only"
            else:
                b_in_q = phrase_hits(q, b_set, b_max)
                b_in_a = phrase_hits(ans_text, b_set, b_max)
                if not (b_in_q or b_in_a):
                    continue                                   # not about any tracked entity
                group = "control"

            # resolve triggering node original spellings + provenance
            trig = {}
            if group in ("f16_only", "both"):
                for ng in (f_in_q | f_in_a):
                    lbl = f_map[ng]
                    trig[lbl] = {"exclusive_to": "f16",
                                 "in_question": ng in f_in_q, "in_answer": ng in f_in_a,
                                 "source_docs": node_source_docs(lbl, ec_f, tc_f)}
            if group in ("q4_only", "both"):
                for ng in (q_in_q | q_in_a):
                    lbl = q_map[ng]
                    trig[lbl] = {"exclusive_to": "q4",
                                 "in_question": ng in q_in_q, "in_answer": ng in q_in_a,
                                 "source_docs": node_source_docs(lbl, ec_q, tc_q)}
            if group == "control":
                for ng in (phrase_hits(q, b_set, b_max) | phrase_hits(ans_text, b_set, b_max)):
                    trig[b_map[ng]] = {"exclusive_to": "both"}

            counts[group] += 1
            fout.write(json.dumps({
                "id": ex.get("id"),
                "size": size, "seed": seed,
                "question": q,
                "answers": golds,
                "context": ex.get("context", ""),
                "group": group,
                "trigger_nodes": trig,
            }, ensure_ascii=False) + "\n")

    print(f"  size={size} v{seed}: f16_only={counts['f16_only']} q4_only={counts['q4_only']} "
          f"both={counts['both']} control={counts['control']}  "
          f"(|f16_only nodes|={len(f16_only)}, |q4_only nodes|={len(q4_only)})")
    return {"size": size, "seed": seed,
            "n_f16_only_nodes": len(f16_only), "n_q4_only_nodes": len(q4_only),
            "n_both_nodes": len(both), **{f"q_{k}": v for k, v in counts.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500, 1000, 2000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--min-label-len", type=int, default=4)
    ap.add_argument("--out-dir", default="node_exclusive_qa")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading MiniLM (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    manifest = []
    for size in args.sizes:
        for seed in args.seeds:
            rec = process_seed(size, seed, model, args)
            if rec:
                manifest.append(rec)

    # aggregate totals
    agg = {}
    for r in manifest:
        a = agg.setdefault(r["size"], {"q_f16_only": 0, "q_q4_only": 0, "q_both": 0, "q_control": 0})
        for k in a:
            a[k] += r[k]
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": manifest, "totals_by_size": agg,
                   "threshold": args.threshold, "min_label_len": args.min_label_len}, f, indent=2)

    print("\n=== TOT(questions) by size ===")
    for size, a in sorted(agg.items()):
        print(f"  size={size}: f16_only={a['q_f16_only']} q4_only={a['q_q4_only']} "
              f"both={a['q_both']} control={a['q_control']}")
    print(f"\nSaved manifest + per-seed subsets in {args.out_dir}/")


if __name__ == "__main__":
    main()

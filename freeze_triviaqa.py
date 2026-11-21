"""
Freeze TriviaQA (rc.wikipedia) for the single-seed F16-vs-Q4 study, sized by
NUMBER OF DOCUMENTS (passages) indexed: 25 / 50 / 75 / 100. NO truncation
(full Wikipedia articles), which is feasible at these small doc counts.

Approach (passage-based, nested across sizes): shuffle questions (seed=0), walk
them accumulating unique evidence documents (entity_pages.wiki_context, title-
prefixed, first-seen order) until we have max(sizes) docs. For each size N we take
the first N docs and keep the questions whose evidence intersects them, with
gold_contexts restricted to the indexed docs. The build script then indexes
exactly those N unique docs.

Output per size: data/triviaqa_docs_{N}.jsonl   (one row per question)
Run:  python freeze_triviaqa.py
"""
import os
import json
import argparse
from datasets import load_dataset

SEED = 0
DATA_DIR = "data"


def example_docs(ex):
    """Ordered unique evidence docs for one question: 'title\\ncontext'."""
    pages = ex.get("entity_pages", {}) or {}
    wiki_ctx = pages.get("wiki_context", []) or []
    titles = pages.get("title", []) or []
    out = []
    for i, ctx in enumerate(wiki_ctx):
        if not ctx:
            continue
        t = titles[i] if i < len(titles) else ""
        doc = f"{t}\n{ctx}".strip() if t else ctx.strip()
        if doc and doc not in out:
            out.append(doc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 75, 100],
                    help="number of DOCUMENTS (passages) to index")
    args = ap.parse_args()
    sizes = sorted(args.sizes)
    maxN = sizes[-1]

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading trivia_qa / rc.wikipedia / train ...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train").shuffle(seed=SEED)

    # Walk questions, accumulate unique docs in first-seen order until maxN docs.
    ordered_docs = []          # unique docs, ordered
    seen = set()
    qa_rows = []               # (id, question, answers, [docs for this q])
    for ex in ds:
        qps = example_docs(ex)
        ans = ex.get("answer", {}) or {}
        qa_rows.append({
            "id": ex.get("question_id"),
            "question": (ex.get("question") or "").strip(),
            "answers": {"value": (ans.get("value") or "").strip(),
                        "aliases": [a.strip() for a in (ans.get("aliases") or []) if a and a.strip()]},
            "qps": qps,
        })
        for d in qps:
            if d not in seen:
                seen.add(d)
                ordered_docs.append(d)
        if len(ordered_docs) >= maxN:
            break

    for N in sizes:
        passN = set(ordered_docs[:N])
        path = os.path.join(DATA_DIR, f"triviaqa_docs_{N}.jsonl")
        n_q = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in qa_rows:
                gold = [d for d in r["qps"] if d in passN]
                if not gold:
                    continue
                n_q += 1
                f.write(json.dumps({"id": r["id"], "question": r["question"],
                                    "answers": r["answers"], "gold_contexts": gold},
                                   ensure_ascii=False) + "\n")
        print(f"docs={N}: {n_q} questions, {len(passN)} indexed docs -> {path}")


if __name__ == "__main__":
    main()

"""
Retrieval-quality metrics for the multi-seed SQuAD KG indexes.

For each question: retrieve-only via rag.aquery_llm (no answer gen, no judge),
then check whether the gold SQuAD context paragraph is among the retrieved
chunks. Reports Hit@k / MRR per version + cross-seed mean.

Gold doc = the question's `context` (the KG was built from these paragraphs).
Recall@k == Hit@k here (one relevant doc per SQuAD question).

IMPORTANT: uses Qwen/Qwen3-Embedding-0.6B to MATCH how the indexes were built.

Usage (per-version array, like the builds/evals):
  sbatch --array=0-4 --export=ALL,SIZE=200,IDX_MODEL=f16 slurm_retrieval_f16.sh
or directly:
  python retrieval_eval.py --size 200 --index-model f16 [--version 0]
"""
import asyncio
import os
import re
import json
import argparse
import numpy as np

from datasets import load_dataset
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True)
parser.add_argument("--index-model", choices=["f16", "q4"], required=True)
parser.add_argument("--version", type=int, default=None, choices=[0, 1, 2, 3, 4])
args = parser.parse_args()

SIZE = args.size
INDEX_MODEL = args.index_model
MODEL_TAG = {"f16": "qwenf16", "q4": "qwenQ4"}[INDEX_MODEL]
VERSIONS = [args.version] if args.version is not None else [0, 1, 2, 3, 4]

# Generator endpoint (used by hybrid mode for query keyword extraction).
# f16 -> :8001, q4 -> :8000 (port overridable via env for parallel jobs).
DEFAULT_PORT = "8001" if INDEX_MODEL == "f16" else "8000"
GEN_BASE_URL = os.environ.get("GEN_BASE_URL", f"http://127.0.0.1:{DEFAULT_PORT}/v1")
API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"

OUT_DIR = "./SQUAD/results"
MODE = "hybrid"
K_LIST = [1, 3, 5, 10]
MAX_K = max(K_LIST)
MIN_CHUNK_CHARS = 40

# -----------------------------
# Generator client
# -----------------------------
gen_client = AsyncOpenAI(base_url=GEN_BASE_URL, api_key=API_KEY)

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await gen_client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.0,
    )
    return resp.choices[0].message.content

# -----------------------------
# Embeddings — MUST match the model the index was built with
# -----------------------------
import torch
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME,
                               device="cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = embedder.get_sentence_embedding_dimension()

@wrap_embedding_func_with_attrs(embedding_dim=EMB_DIM, max_token_size=8192,
                                model_name=EMBEDDING_MODEL_NAME)
async def embedding_func(texts):
    vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)

# -----------------------------
# Matching helpers
# -----------------------------
def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def chunk_matches(chunk_text, gold_context):
    c, g = _norm(chunk_text), _norm(gold_context)
    if len(c) < MIN_CHUNK_CHARS:
        return False
    return (c in g) or (g in c)

def extract_chunks(result):
    data = result.get("data", {}) if isinstance(result, dict) else {}
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    return [(ch.get("content", "") if isinstance(ch, dict) else str(ch)) for ch in chunks]

def first_match_rank(chunks, gold_context):
    for idx, t in enumerate(chunks, start=1):
        if chunk_matches(t, gold_context):
            return idx
    return None

# -----------------------------
# One version
# -----------------------------
async def eval_version(version):
    workdir = f"./SQUAD/{INDEX_MODEL}/{SIZE}/v{version}"
    data_path = f"./data/squad_train_{SIZE}_v{version}.jsonl"
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Index dir not found: {workdir}")

    rag = LightRAG(working_dir=workdir, llm_model_func=llm_model_func,
                   llm_model_name=MODEL_NAME, embedding_func=embedding_func,
                   llm_model_max_async=1, embedding_func_max_async=4)
    await rag.initialize_storages()
    ds = load_dataset("json", data_files=data_path, split="train")

    hit = {k: 0 for k in K_LIST}
    rr_sum = 0.0
    empty = 0
    n = len(ds)
    for i, ex in enumerate(ds):
        param = QueryParam(mode=MODE, top_k=MAX_K, chunk_top_k=MAX_K,
                           enable_rerank=False, stream=False)
        result = await rag.aquery_llm(ex["question"], param=param)
        chunks = extract_chunks(result)[:MAX_K]
        if not chunks:
            empty += 1
        r = first_match_rank(chunks, ex["context"])
        if r is not None:
            rr_sum += 1.0 / r
        for k in K_LIST:
            hit[k] += int(r is not None and r <= k)
        print(f"  [v{version}] [{i+1}/{n}] rank={r}")

    res = {"version": version, "n": n, "empty_chunks": empty,
           "mrr": rr_sum / n if n else 0.0}
    for k in K_LIST:
        res[f"hit@{k}"] = hit[k] / n if n else 0.0
    with open(os.path.join(OUT_DIR, f"retrieval_{SIZE}_v{version}_{MODEL_TAG}.json"),
              "w", encoding="utf-8") as jf:
        json.dump(res, jf, indent=2)
    print(f"  v{version}: " + " ".join(f"hit@{k}={res[f'hit@{k}']:.3f}" for k in K_LIST)
          + f" mrr={res['mrr']:.3f}")
    return res

# -----------------------------
# Main + cross-seed summary (rebuilt from per-version JSONs on disk)
# -----------------------------
async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for v in VERSIONS:
        await eval_version(v)

    results = []
    for v in range(5):
        p = os.path.join(OUT_DIR, f"retrieval_{SIZE}_v{v}_{MODEL_TAG}.json")
        if os.path.exists(p):
            with open(p) as jf:
                results.append(json.load(jf))
    if not results:
        return
    results.sort(key=lambda r: r["version"])
    keys = [f"hit@{k}" for k in K_LIST] + ["mrr"]
    summ = os.path.join(OUT_DIR, f"retrieval_{SIZE}_{MODEL_TAG}_summary.txt")
    with open(summ, "w", encoding="utf-8") as f:
        f.write(f"=== RETRIEVAL SUMMARY: SQuAD size={SIZE}, model={MODEL_TAG} (mode={MODE}) ===\n")
        f.write(f"Seeds present: {len(results)}/5\n\n")
        for r in results:
            f.write(f"v{r['version']} (n={r['n']}): "
                    + " ".join(f"{k}={r[k]:.4f}" for k in keys)
                    + f"  empty={r['empty_chunks']}\n")
        f.write("\n")
        for k in keys:
            vals = [r[k] for r in results]
            f.write(f"Mean {k}: {np.mean(vals):.4f} (std {np.std(vals):.4f})\n")
    print(f"Saved {summ}")


if __name__ == "__main__":
    asyncio.run(main())

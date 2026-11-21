"""
Retrieval-quality metrics for the TriviaQA KG indexes (single seed).

Like retrieval_eval.py but: single seed, doc-count sizes 25/50/75/100, and the
gold target is the question's list of `gold_contexts` (full Wikipedia articles) --
a retrieved chunk counts as a hit if it is a substring of ANY gold context (or a
gold context contains it). Reports Hit@k / MRR per size. Needs the generator
server (hybrid mode extracts query keywords with the LLM).

Usage:
  python retrieval_eval_triviaqa.py --index-model f16 [--size 25]
"""
import asyncio, os, re, json, argparse
import numpy as np
from datasets import load_dataset
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import torch

from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs

parser = argparse.ArgumentParser()
parser.add_argument("--index-model", choices=["f16", "q4"], required=True)
parser.add_argument("--size", type=int, default=None, choices=[25, 50, 75, 100])
args = parser.parse_args()

INDEX_MODEL = args.index_model
MODEL_TAG = {"f16": "qwenf16", "q4": "qwenQ4"}[INDEX_MODEL]
SIZES = [args.size] if args.size is not None else [25, 50, 75, 100]

DEFAULT_PORT = "8001" if INDEX_MODEL == "f16" else "8000"
GEN_BASE_URL = os.environ.get("GEN_BASE_URL", f"http://127.0.0.1:{DEFAULT_PORT}/v1")
API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"
OUT_DIR = "./TRIVIAQA/results"
MODE = "hybrid"
K_LIST = [1, 3, 5, 10]
MAX_K = max(K_LIST)
MIN_CHUNK_CHARS = 40

gen_client = AsyncOpenAI(base_url=GEN_BASE_URL, api_key=API_KEY)

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await gen_client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.0)
    return resp.choices[0].message.content

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME,
                               device="cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = embedder.get_sentence_embedding_dimension()

@wrap_embedding_func_with_attrs(embedding_dim=EMB_DIM, max_token_size=8192,
                                model_name=EMBEDDING_MODEL_NAME)
async def embedding_func(texts):
    vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def chunk_matches_any(chunk_text, gold_contexts_norm):
    c = _norm(chunk_text)
    if len(c) < MIN_CHUNK_CHARS:
        return False
    return any((c in g) or (g in c) for g in gold_contexts_norm)

def extract_chunks(result):
    data = result.get("data", {}) if isinstance(result, dict) else {}
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    return [(ch.get("content", "") if isinstance(ch, dict) else str(ch)) for ch in chunks]

def first_match_rank(chunks, gold_contexts_norm):
    for idx, t in enumerate(chunks, start=1):
        if chunk_matches_any(t, gold_contexts_norm):
            return idx
    return None


async def eval_size(size):
    workdir = f"./TRIVIAQA/{INDEX_MODEL}/{size}"
    data_path = f"./data/triviaqa_docs_{size}.jsonl"
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Index dir not found: {workdir}")
    rag = LightRAG(working_dir=workdir, llm_model_func=llm_model_func, llm_model_name=MODEL_NAME,
                   embedding_func=embedding_func, llm_model_max_async=1, embedding_func_max_async=4)
    await rag.initialize_storages()
    ds = load_dataset("json", data_files=data_path, split="train")

    hit = {k: 0 for k in K_LIST}
    rr_sum, empty, n = 0.0, 0, len(ds)
    for i, ex in enumerate(ds):
        golds_norm = [_norm(g) for g in (ex.get("gold_contexts") or []) if g]
        param = QueryParam(mode=MODE, top_k=MAX_K, chunk_top_k=MAX_K, enable_rerank=False, stream=False)
        result = await rag.aquery_llm(ex["question"], param=param)
        chunks = extract_chunks(result)[:MAX_K]
        if not chunks:
            empty += 1
        r = first_match_rank(chunks, golds_norm) if golds_norm else None
        if r is not None:
            rr_sum += 1.0 / r
        for k in K_LIST:
            hit[k] += int(r is not None and r <= k)
        print(f"  [{size}] [{i+1}/{n}] rank={r}")

    res = {"size": size, "n": n, "empty_chunks": empty, "mrr": rr_sum / n if n else 0.0}
    for k in K_LIST:
        res[f"hit@{k}"] = hit[k] / n if n else 0.0
    with open(os.path.join(OUT_DIR, f"retrieval_triviaqa_{size}_{MODEL_TAG}.json"), "w") as jf:
        json.dump(res, jf, indent=2)
    print(f"  size {size}: " + " ".join(f"hit@{k}={res[f'hit@{k}']:.3f}" for k in K_LIST) + f" mrr={res['mrr']:.3f}")
    return res


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SIZES:
        await eval_size(s)
    results = []
    for s in [25, 50, 75, 100]:
        p = os.path.join(OUT_DIR, f"retrieval_triviaqa_{s}_{MODEL_TAG}.json")
        if os.path.exists(p):
            results.append(json.load(open(p)))
    if not results:
        return
    keys = [f"hit@{k}" for k in K_LIST] + ["mrr"]
    summ = os.path.join(OUT_DIR, f"retrieval_triviaqa_{MODEL_TAG}_summary.txt")
    with open(summ, "w") as f:
        f.write(f"=== TriviaQA RETRIEVAL SUMMARY: model={MODEL_TAG} (mode={MODE}) ===\n")
        f.write(f"Sizes present: {len(results)}/4\n\n")
        for r in results:
            f.write(f"size {r['size']} (n={r['n']}): " + " ".join(f"{k}={r[k]:.4f}" for k in keys) + f"  empty={r['empty_chunks']}\n")
    print(f"Saved {summ}")


if __name__ == "__main__":
    asyncio.run(main())

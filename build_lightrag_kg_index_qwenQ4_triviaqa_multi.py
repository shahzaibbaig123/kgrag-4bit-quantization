# build_lightrag_kg_index_qwenQ4_triviaqa_multi.py
"""
Build the Q4 LightRAG KG index for TriviaQA (single seed), one corpus size.
Identical to the F16 builder except INDEX_MODEL=q4 and the default generator port (8000).
Output: TRIVIAQA/q4/{size}/

Usage (server started by slurm_build_Q4.sh on :8000; LLM_BASE_URL overridable):
  python build_lightrag_kg_index_qwenQ4_triviaqa_multi.py --size 500
"""
import os
import json
import shutil
import asyncio
import argparse
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True, choices=[25, 50, 75, 100],
                    help="number of DOCUMENTS (passages) to index")
parser.add_argument("--resume", action="store_true",
                    help="do NOT wipe the workdir; re-process FAILED/PENDING docs, keep PROCESSED")
args = parser.parse_args()
SIZE = args.size

# Q4 generator. Port overridable via env for parallel jobs.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"

INDEX_MODEL = "q4"
CHUNK_TOKEN_SIZE = 300
CHUNK_OVERLAP_TOKEN_SIZE = 50
# Cap extraction output. Greedy decoding (temp=0) on noisy Wikipedia chunks can run away into a
# repetition loop that fills the whole 32k context; the gleaning pass then resends that response
# as history -> request exceeds the context -> 400 -> doc FAILED. A legit extraction of a 300-tok
# chunk is ~1-1.5k tokens, so 4096 is generous headroom while making the runaway-400 impossible.
GEN_MAX_TOKENS = 4096
LLM_MAX_ASYNC = 4
EMB_MAX_ASYNC = 4
MAX_PARALLEL_INSERT = 2

DATA_PATH = f"./data/triviaqa_docs_{SIZE}.jsonl"
WORKDIR = f"./TRIVIAQA/{INDEX_MODEL}/{SIZE}"


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # Retry transient call errors so a single hiccup can't cascade into a whole-doc failure.
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=GEN_MAX_TOKENS)
            return resp.choices[0].message.content
        except Exception as e:
            last = e
            await asyncio.sleep(2 * (attempt + 1))
    raise last


EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME,
                               device="cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = embedder.get_sentence_embedding_dimension()


@wrap_embedding_func_with_attrs(embedding_dim=EMB_DIM, max_token_size=8192,
                                model_name=EMBEDDING_MODEL_NAME)
async def embedding_func(texts):
    vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def load_corpus_and_write_manifest(data_path, workdir):
    seen, docs = set(), []
    n_q = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            n_q += 1
            for d in json.loads(line).get("gold_contexts", []):
                if d and d not in seen:
                    seen.add(d)
                    docs.append(d)
    manifest = {
        "dataset": "triviaqa_rc_wikipedia", "size_questions": n_q,
        "n_unique_evidence_docs_indexed": len(docs),
        "data_path": data_path, "llm_base_url": LLM_BASE_URL,
        "chunk_token_size": CHUNK_TOKEN_SIZE, "chunk_overlap_token_size": CHUNK_OVERLAP_TOKEN_SIZE,
    }
    with open(os.path.join(workdir, "indexed_slice_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return docs


async def main():
    print(f"Building TriviaQA size={SIZE} ({INDEX_MODEL}) LLM={LLM_BASE_URL}")
    print(f"  Data:   {DATA_PATH}")
    print(f"  Output: {WORKDIR}")
    if args.resume and os.path.isdir(WORKDIR):
        print("  RESUME: keeping existing index; LightRAG will re-process FAILED/PENDING docs.")
    else:
        reset_dir(WORKDIR)

    rag = LightRAG(
        working_dir=WORKDIR,
        llm_model_func=llm_model_func,
        llm_model_name=MODEL_NAME,
        embedding_func=embedding_func,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        llm_model_max_async=LLM_MAX_ASYNC,
        embedding_func_max_async=EMB_MAX_ASYNC,
        max_parallel_insert=MAX_PARALLEL_INSERT,
    )
    await rag.initialize_storages()

    docs = load_corpus_and_write_manifest(DATA_PATH, WORKDIR)
    await rag.ainsert(docs)
    print(f"Done: {WORKDIR}  (docs inserted: {len(docs)})")


if __name__ == "__main__":
    asyncio.run(main())

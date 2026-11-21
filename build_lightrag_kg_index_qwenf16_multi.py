"""
Build the F16 LightRAG KG index for SQuAD, one corpus size, all five seeds (or one via --version).
Reads data/squad_train_{size}_v{seed}.jsonl and indexes its unique context paragraphs.
Output: SQUAD/f16/{size}/v{seed}/

Usage (server started by slurm_build_f16.sh on :8001; LLM_BASE_URL overridable):
  python build_lightrag_kg_index_qwenf16_multi.py --size 500 [--version 0]
"""
import os
import shutil
import asyncio
import json
import argparse
import numpy as np
import torch

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs


# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True, choices=[200, 500, 1000, 2000, 3000, 4000, 8000])
parser.add_argument("--version", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="Build only this seed. If omitted, builds all 5 (v0-v4).")
args = parser.parse_args()

SIZE = args.size
# If --version is given, build just that seed; otherwise all five.
VERSIONS = [args.version] if args.version is not None else [0, 1, 2, 3, 4]

# llama-server running F16 model (port overridable via env for parallel jobs)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
LLM_API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"

CHUNK_TOKEN_SIZE = 600
CHUNK_OVERLAP_TOKEN_SIZE = 100
LLM_MAX_ASYNC = 4
# Documents processed concurrently. SQuAD = 1 chunk/doc, so this is effectively
# the number of concurrent extraction LLM calls -> match the server's slots.
MAX_PARALLEL_INSERT = 4
EMB_MAX_ASYNC = 4


# -----------------------------
# Helpers
# -----------------------------
def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# -----------------------------
# LLM client
# -----------------------------
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content


# -----------------------------
# Embeddings
# -----------------------------
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

embedder = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

EMB_DIM = embedder.get_sentence_embedding_dimension()


@wrap_embedding_func_with_attrs(
    embedding_dim=EMB_DIM,
    max_token_size=8192,
    model_name=EMBEDDING_MODEL_NAME,
)
async def embedding_func(texts):
    vecs = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


# -----------------------------
# Corpus loader
# -----------------------------
def load_corpus_and_write_manifest(data_path, workdir, version):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=data_path, split="train")

    seen = set()
    contexts = []
    for ex in ds:
        c = ex["context"]
        if c not in seen:
            seen.add(c)
            contexts.append(c)

    manifest = {
        "data_path": data_path,
        "size": SIZE,
        "version": version,
        "n_examples_in_file": len(ds),
        "n_unique_contexts_indexed": len(contexts),
    }
    manifest_path = os.path.join(workdir, "indexed_slice_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return contexts


# -----------------------------
# Build one version
# -----------------------------
async def build_version(version: int):
    workdir = f"./SQUAD/f16/{SIZE}/v{version}"
    data_path = f"./data/squad_train_{SIZE}_v{version}.jsonl"

    print(f"\n--- size={SIZE}, version={version} ---")
    print(f"  Data:   {data_path}")
    print(f"  Output: {workdir}")

    reset_dir(workdir)

    rag = LightRAG(
        working_dir=workdir,
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

    docs = load_corpus_and_write_manifest(data_path, workdir, version)
    # Batch insert: pass the whole list so LightRAG pipelines extraction across
    # chunks with llm_model_max_async concurrency (instead of one doc at a time).
    await rag.ainsert(docs)

    print(f"Done: {workdir}")


# -----------------------------
# Main
# -----------------------------
async def main():
    print(f"Building size={SIZE}, versions={VERSIONS} (LLM={LLM_BASE_URL})")
    for v in VERSIONS:
        await build_version(v)
    print(f"\nComplete for size={SIZE}, versions={VERSIONS}.")


if __name__ == "__main__":
    asyncio.run(main())

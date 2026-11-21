"""
SQuAD correctness eval (F16), all five seeds (or one via --version), sizes 200-3000.

Loads the pre-built KG at SQUAD/f16/{size}/v{seed}/, answers each question in hybrid
mode (top_k=5), and scores it 1-5 with the LlamaIndex CorrectnessEvaluator (gpt-oss-20b
judge, pass >= 4). Output: SQUAD/results/eval_squad_{size}_v{seed}_qwenf16.{txt,json}
plus a self-healing cross-seed summary.

Needs two llama-servers: generator (Qwen F16, :8001) + judge (gpt-oss-20b, :8005).
Usage:  python eval_squad_multi_f16.py --size 500 [--version 0]
"""
import asyncio
import argparse
import os
import json
import numpy as np
import torch
from datasets import load_dataset
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs

from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI as LIOpenAI


# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True, choices=[200, 500, 1000, 2000, 3000, 4000, 8000])
parser.add_argument("--version", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="Eval only this seed. If omitted, evaluates all 5 (v0-v4).")
args = parser.parse_args()

SIZE = args.size
VERSIONS = [args.version] if args.version is not None else [0, 1, 2, 3, 4]

# Model / index identity (F16)
MODEL_TAG = "qwenf16"      # used in output filenames
INDEX_MODEL = "f16"        # ./SQUAD/{INDEX_MODEL}/{size}/v{n}

# Answering model (F16) server
GEN_BASE_URL = "http://127.0.0.1:8001/v1"
# Judge model (F16) server
JUDGE_BASE_URL = "http://127.0.0.1:8005/v1"

API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"  # placeholder; llama-server ignores it, LlamaIndex validates it

OUT_DIR = "./SQUAD/results"

MODE = "hybrid"
TOP_K = 5
PASS_THRESHOLD = 4.0
GEN_MAX_TOKENS = 512


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
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=GEN_MAX_TOKENS,
    )
    return resp.choices[0].message.content


# -----------------------------
# Embeddings (Qwen3-Embedding-0.6B) — loaded once, reused for all versions
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
# Correctness judge
# -----------------------------
judge_llm = LIOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    api_base=JUDGE_BASE_URL,
)
correctness = CorrectnessEvaluator(llm=judge_llm)


# -----------------------------
# Evaluate one version
# -----------------------------
async def eval_version(version: int):
    workdir = f"./SQUAD/{INDEX_MODEL}/{SIZE}/v{version}"
    data_path = f"./data/squad_train_{SIZE}_v{version}.jsonl"
    out_file = os.path.join(OUT_DIR, f"eval_squad_{SIZE}_v{version}_{MODEL_TAG}.txt")

    print(f"\n=== size={SIZE}, version={version} ({INDEX_MODEL}) ===")
    print(f"  Index: {workdir}")
    print(f"  Data:  {data_path}")

    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Index dir not found: {workdir} (build it first)")

    rag = LightRAG(
        working_dir=workdir,
        llm_model_func=llm_model_func,
        llm_model_name=MODEL_NAME,
        embedding_func=embedding_func,
        llm_model_max_async=1,
        embedding_func_max_async=4,
    )
    await rag.initialize_storages()

    ds = load_dataset("json", data_files=data_path, split="train")

    scores = []
    passed = 0
    processed = 0
    skipped = 0
    failed = 0
    total = len(ds)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation: Frozen SQuAD size={SIZE} v{version} ({MODEL_TAG})\n")
        f.write(f"Mode: {MODE}\n")
        f.write(f"Top-K: {TOP_K}\n")
        f.write(f"Judge: gpt-oss-20b-F16\n\n")

        for i, ex in enumerate(ds):
            q = ex.get("question", "").strip()
            answers = ex.get("answers", {}).get("text", [])

            if not q:
                skipped += 1
                f.write(f"#{i+1}\n")
                f.write("ERROR: Empty question\n")
                f.write("-" * 80 + "\n")
                print(f"  [v{version}] [{i+1}/{total}] skipped: empty question")
                continue

            if not answers:
                skipped += 1
                f.write(f"#{i+1}\n")
                f.write(f"Q: {q}\n")
                f.write("ERROR: No reference answer found\n")
                f.write("-" * 80 + "\n")
                print(f"  [v{version}] [{i+1}/{total}] skipped: no reference answer")
                continue

            ref = str(answers[0]).strip()

            try:
                resp = await rag.aquery(
                    q,
                    param=QueryParam(mode=MODE, top_k=TOP_K, enable_rerank=False),
                )
                answer_text = str(resp).strip()
            except Exception as e:
                failed += 1
                f.write(f"#{i+1}\n")
                f.write(f"Q: {q}\n")
                f.write(f"Gold: {ref}\n")
                f.write(f"ERROR during retrieval: {e}\n")
                f.write("-" * 80 + "\n")
                print(f"  [v{version}] [{i+1}/{total}] retrieval failed: {e}")
                continue

            try:
                result = await correctness.aevaluate(
                    query=q,
                    response=answer_text,
                    reference=ref,
                )

                if result.score is None:
                    raise ValueError("Evaluator returned no score")

                s = float(result.score)
                is_pass = s >= PASS_THRESHOLD
                scores.append(s)
                passed += int(is_pass)
                processed += 1

                f.write(f"#{i+1}\n")
                f.write(f"Q: {q}\n")
                f.write(f"Gold: {ref}\n")
                f.write(f"Answer: {answer_text}\n")
                f.write(f"Score: {s:.1f} | Pass: {is_pass}\n")
                f.write("-" * 80 + "\n")

                print(f"  [v{version}] [{i+1}/{total}] score={s:.1f} pass={is_pass}")

            except Exception as e:
                failed += 1
                f.write(f"#{i+1}\n")
                f.write(f"Q: {q}\n")
                f.write(f"Gold: {ref}\n")
                f.write(f"Answer: {answer_text}\n")
                f.write(f"ERROR during evaluation: {e}\n")
                f.write("-" * 80 + "\n")
                print(f"  [v{version}] [{i+1}/{total}] eval failed: {e}")
                continue

        avg = sum(scores) / len(scores) if scores else 0.0
        pass_rate = passed / len(scores) if scores else 0.0

        f.write("\n=== SUMMARY ===\n")
        f.write(f"Total samples in dataset: {total}\n")
        f.write(f"Processed successfully: {processed}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Average correctness: {avg:.4f}\n")
        f.write(f"Pass@{PASS_THRESHOLD}: {passed}/{processed} = {pass_rate:.4f}\n")

    result = {
        "version": version,
        "n": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "avg": avg,
        "pass_rate": pass_rate,
        "passed": passed,
    }
    # per-version JSON so the cross-seed summary can be rebuilt from disk
    # (works whether versions run all-at-once or as separate array jobs)
    with open(os.path.join(OUT_DIR, f"eval_squad_{SIZE}_v{version}_{MODEL_TAG}.json"),
              "w", encoding="utf-8") as jf:
        json.dump(result, jf, indent=2)

    print(f"  Saved: {out_file}")
    return result


# -----------------------------
# Main: loop all versions + cross-seed summary
# -----------------------------
async def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for v in VERSIONS:
        await eval_version(v)

    # Rebuild the cross-seed summary from ALL per-version JSONs present on disk,
    # so it is correct whether versions ran together or as separate array jobs.
    results = []
    for v in [0, 1, 2, 3, 4]:
        p = os.path.join(OUT_DIR, f"eval_squad_{SIZE}_v{v}_{MODEL_TAG}.json")
        if os.path.exists(p):
            with open(p) as jf:
                results.append(json.load(jf))
    if not results:
        print("No per-version results to summarize.")
        return
    results.sort(key=lambda r: r["version"])
    avgs = [r["avg"] for r in results]
    pass_rates = [r["pass_rate"] for r in results]

    summary_file = os.path.join(OUT_DIR, f"eval_squad_{SIZE}_{MODEL_TAG}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"=== CROSS-SEED SUMMARY: SQuAD size={SIZE}, model={MODEL_TAG} ===\n")
        f.write(f"Mode: {MODE} | Top-K: {TOP_K} | Pass threshold: {PASS_THRESHOLD}\n")
        f.write(f"Seeds present: {len(results)}/5\n\n")
        for r in results:
            f.write(
                f"v{r['version']}: avg={r['avg']:.4f}  "
                f"pass@{PASS_THRESHOLD}={r['passed']}/{r['processed']}={r['pass_rate']:.4f}  "
                f"(skipped={r['skipped']}, failed={r['failed']})\n"
            )
        f.write("\n")
        f.write(f"Mean avg correctness: {np.mean(avgs):.4f} (std {np.std(avgs):.4f})\n")
        f.write(f"Mean pass@{PASS_THRESHOLD}: {np.mean(pass_rates):.4f} (std {np.std(pass_rates):.4f})\n")

    # machine-readable copy
    json_file = os.path.join(OUT_DIR, f"eval_squad_{SIZE}_{MODEL_TAG}_summary.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "size": SIZE,
                "model": MODEL_TAG,
                "mode": MODE,
                "top_k": TOP_K,
                "pass_threshold": PASS_THRESHOLD,
                "per_version": results,
                "mean_avg": float(np.mean(avgs)),
                "std_avg": float(np.std(avgs)),
                "mean_pass_rate": float(np.mean(pass_rates)),
                "std_pass_rate": float(np.std(pass_rates)),
            },
            f,
            indent=2,
        )

    print(f"\nSaved cross-seed summary: {summary_file}")
    print(f"Saved JSON summary:       {json_file}")


if __name__ == "__main__":
    asyncio.run(main())

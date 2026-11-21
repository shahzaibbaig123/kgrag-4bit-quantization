"""
TriviaQA correctness eval (Q4), single seed, doc-count sizes 25/50/75/100.

Identical to eval_triviaqa_f16.py except: Q4 generator (:8000), second judge
instance (:8004), index TRIVIAQA/q4/{size}/, tag qwenQ4.
Usage:  python eval_triviaqa_q4.py [--size 25]
"""
import asyncio, argparse, os, json
import numpy as np
import torch
from datasets import load_dataset
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI as LIOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=None, choices=[25, 50, 75, 100],
                    help="Eval only this size. If omitted, evaluates all four.")
args = parser.parse_args()
SIZES = [args.size] if args.size is not None else [25, 50, 75, 100]

MODEL_TAG = "qwenQ4"
INDEX_MODEL = "q4"
GEN_BASE_URL = os.environ.get("GEN_BASE_URL", "http://127.0.0.1:8000/v1")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8004/v1")
API_KEY = "no-key"
MODEL_NAME = "gpt-3.5-turbo"
OUT_DIR = "./TRIVIAQA/results"
MODE = "hybrid"
TOP_K = 5
PASS_THRESHOLD = 4.0
GEN_MAX_TOKENS = 512

gen_client = AsyncOpenAI(base_url=GEN_BASE_URL, api_key=API_KEY)

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await gen_client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=GEN_MAX_TOKENS)
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

judge_llm = LIOpenAI(model=MODEL_NAME, api_key=API_KEY, api_base=JUDGE_BASE_URL)
correctness = CorrectnessEvaluator(llm=judge_llm)


async def eval_size(size: int):
    workdir = f"./TRIVIAQA/{INDEX_MODEL}/{size}"
    data_path = f"./data/triviaqa_docs_{size}.jsonl"
    out_file = os.path.join(OUT_DIR, f"eval_triviaqa_{size}_{MODEL_TAG}.txt")
    print(f"\n=== TriviaQA size={size} ({INDEX_MODEL}) ===\n  Index: {workdir}\n  Data:  {data_path}")
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Index dir not found: {workdir} (build it first)")

    rag = LightRAG(working_dir=workdir, llm_model_func=llm_model_func, llm_model_name=MODEL_NAME,
                   embedding_func=embedding_func, llm_model_max_async=1, embedding_func_max_async=4)
    await rag.initialize_storages()
    ds = load_dataset("json", data_files=data_path, split="train")

    scores, passed, processed, skipped, failed = [], 0, 0, 0, 0
    total = len(ds)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation: Frozen TriviaQA size={size} ({MODEL_TAG})\nMode: {MODE}\nTop-K: {TOP_K}\nJudge: gpt-oss-20b\n\n")
        for i, ex in enumerate(ds):
            q = (ex.get("question") or "").strip()
            ref = str(((ex.get("answers") or {}).get("value") or "")).strip()
            if not q:
                skipped += 1; f.write(f"#{i+1}\nERROR: Empty question\n" + "-" * 80 + "\n"); continue
            if not ref:
                skipped += 1; f.write(f"#{i+1}\nQ: {q}\nERROR: No reference answer found\n" + "-" * 80 + "\n"); continue
            try:
                resp = await rag.aquery(q, param=QueryParam(mode=MODE, top_k=TOP_K, enable_rerank=False))
                answer_text = str(resp).strip()
            except Exception as e:
                failed += 1; f.write(f"#{i+1}\nQ: {q}\nGold: {ref}\nERROR during retrieval: {e}\n" + "-" * 80 + "\n")
                print(f"  [{i+1}/{total}] retrieval failed: {e}"); continue
            try:
                result = await correctness.aevaluate(query=q, response=answer_text, reference=ref)
                if result.score is None:
                    raise ValueError("Evaluator returned no score")
                s = float(result.score); is_pass = s >= PASS_THRESHOLD
                scores.append(s); passed += int(is_pass); processed += 1
                f.write(f"#{i+1}\nQ: {q}\nGold: {ref}\nAnswer: {answer_text}\nScore: {s:.1f} | Pass: {is_pass}\n" + "-" * 80 + "\n")
                print(f"  [{i+1}/{total}] score={s:.1f} pass={is_pass}")
            except Exception as e:
                failed += 1; f.write(f"#{i+1}\nQ: {q}\nGold: {ref}\nAnswer: {answer_text}\nERROR during evaluation: {e}\n" + "-" * 80 + "\n")
                print(f"  [{i+1}/{total}] eval failed: {e}"); continue

        avg = sum(scores) / len(scores) if scores else 0.0
        pass_rate = passed / len(scores) if scores else 0.0
        f.write(f"\n=== SUMMARY ===\nTotal samples in dataset: {total}\nProcessed successfully: {processed}\n"
                f"Skipped: {skipped}\nFailed: {failed}\nAverage correctness: {avg:.4f}\n"
                f"Pass@{PASS_THRESHOLD}: {passed}/{processed} = {pass_rate:.4f}\n")

    res = {"size": size, "n": total, "processed": processed, "skipped": skipped,
           "failed": failed, "avg": avg, "pass_rate": pass_rate, "passed": passed}
    with open(os.path.join(OUT_DIR, f"eval_triviaqa_{size}_{MODEL_TAG}.json"), "w") as jf:
        json.dump(res, jf, indent=2)
    print(f"  Saved: {out_file}")
    return res


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SIZES:
        await eval_size(s)
    results = []
    for s in [25, 50, 75, 100]:
        p = os.path.join(OUT_DIR, f"eval_triviaqa_{s}_{MODEL_TAG}.json")
        if os.path.exists(p):
            results.append(json.load(open(p)))
    if not results:
        return
    summary = os.path.join(OUT_DIR, f"eval_triviaqa_{MODEL_TAG}_summary.txt")
    with open(summary, "w") as f:
        f.write(f"=== TriviaQA CORRECTNESS SUMMARY: model={MODEL_TAG} (mode={MODE}, top_k={TOP_K}) ===\n")
        f.write(f"Sizes present: {len(results)}/4\n\n")
        for r in results:
            f.write(f"size {r['size']}: avg={r['avg']:.4f}  pass@{PASS_THRESHOLD}={r['passed']}/{r['processed']}={r['pass_rate']:.4f}  (skipped={r['skipped']}, failed={r['failed']})\n")
    print(f"\nSaved {summary}")


if __name__ == "__main__":
    asyncio.run(main())

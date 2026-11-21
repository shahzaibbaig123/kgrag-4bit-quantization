#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/buildtrvi_Q4_%x_%j.out
#SBATCH --job-name=buildtrvi_Q4

# TriviaQA Q4 KG build (single seed), one DOC-count size (25/50/75/100).
# Runs on A100 (AMD nodes): AVX2-safe rag-a100 venv + build_a100 llama-server.
#   sbatch --job-name=btrvi_q4_25 --export=ALL,SIZE=25,RESUME=1 slurm_build_triviaqa_q4.sh

if [ -z "$SIZE" ]; then echo "ERROR: SIZE not set (25|50|75|100)"; exit 1; fi

# AMD_A100 software stack + the AVX2-safe venv (Intel rag-py311 faults on AMD nodes)
module use /software/easybuild-AMD_A100/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source ~/venvs/rag-a100/bin/activate
mkdir -p logs

# job-unique port so parallel/co-located builds never collide
PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

# llama.cpp built for A100 (build_a100), Q4 model
./llama.cpp/build_a100/bin/llama-server -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port "$PORT" -ngl 99 -c 32768 -t 8 &
LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null' EXIT

echo "Waiting for llama-server on :$PORT ..."
for attempt in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; echo "ready"; break; fi
    [ "$attempt" -eq 180 ] && { echo "ERROR: server not ready"; exit 1; }
    sleep 5
done

# Timeout headroom for long Wikipedia articles. LightRAG's per-task health-check kills a
# worker at ~(timeout*2 + 15)s; defaults (LLM 180 -> 375s, EMBEDDING 30 -> 75s) were too low and
# the CPU-side embedder on the longest articles took >75s, cascading whole docs to FAILED.
export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"             # -> ~615s LLM worker hard kill
export EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-300}" # -> ~615s embedder worker hard kill (was 75s)
RESUME_ARG=""; [ -n "$RESUME" ] && RESUME_ARG="--resume"
python build_lightrag_kg_index_qwenQ4_triviaqa_multi.py --size "$SIZE" $RESUME_ARG
exit $?

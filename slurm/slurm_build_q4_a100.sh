#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/build_q4_a100_%x_%j.out
#SBATCH --job-name=build_q4_a100

# Per-seed Q4 build on A100 (AMD nodes) using the rag-a100 venv.
# VERSION comes from the array task id (preferred) or an explicit VERSION env.
#
# Short queue (parallel seeds, 8h):
#   sbatch --array=0-4 --export=ALL,SIZE=200 slurm_build_q4_a100.sh
# Medium queue (1 at a time, 24h):
#   sbatch -p A100medium -t 24:00:00 --array=0-4%1 --export=ALL,SIZE=4000 slurm_build_q4_a100.sh

VERSION="${VERSION:-$SLURM_ARRAY_TASK_ID}"
if [ -z "$SIZE" ] || [ -z "$VERSION" ]; then
    echo "ERROR: set SIZE and a seed (via --array=0-4 or VERSION=N)"
    exit 1
fi

# AMD_A100 software stack + the AVX2-safe venv
module use /software/easybuild-AMD_A100/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source ~/venvs/rag-a100/bin/activate

mkdir -p logs

# Job-unique port so jobs sharing a node never collide.
PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

trap 'kill $LLAMA_PID 2>/dev/null' EXIT

# llama.cpp built for A100 (build_a100), Q4 model
./llama.cpp/build_a100/bin/llama-server \
    -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --port "$PORT" \
    -ngl 99 \
    -c 16384 \
    -t 8 \
    &
LLAMA_PID=$!

echo "Waiting for llama-server on :$PORT ..."
for attempt in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
        exec 3>&-
        echo "llama-server :$PORT ready."
        break
    fi
    if [ "$attempt" -eq 180 ]; then
        echo "ERROR: llama-server :$PORT not ready after 15 min; aborting."
        exit 1
    fi
    sleep 5
done

python build_lightrag_kg_index_qwenQ4_multi.py --size "$SIZE" --version "$VERSION"
EXIT_CODE=$?

exit $EXIT_CODE

#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/build_f16_short_%x_%j.out
#SBATCH --job-name=build_f16_short

# Per-seed F16 build. One job = one seed; seeds can run in parallel.
# VERSION comes from the array task id (preferred) or an explicit VERSION env.
#
# Short queue (fits <8h, runs all seeds in parallel):
#   sbatch --array=0-4 --export=ALL,SIZE=200 slurm_build_f16_short.sh
# Medium queue (>8h per seed, runs 1 at a time via %1):
#   sbatch -p A40medium -t 24:00:00 --array=0-4%1 --export=ALL,SIZE=2000 slurm_build_f16_short.sh

# Prefer the array task id; fall back to an explicit VERSION.
VERSION="${VERSION:-$SLURM_ARRAY_TASK_ID}"

if [ -z "$SIZE" ] || [ -z "$VERSION" ]; then
    echo "ERROR: set SIZE and a seed (via --array=0-4 or VERSION=N)"
    exit 1
fi

source ~/venvs/rag-py311/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load CUDA/12.4.0

mkdir -p logs

# Job-unique port (derived from the job id) so any jobs sharing a node never
# collide on the same port, regardless of size/seed.
PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"

trap 'kill $LLAMA_PID 2>/dev/null' EXIT

# Start llama.cpp server (F16 model) on the per-seed port
./llama.cpp/build/bin/llama-server \
    -m ~/models/Qwen2.5-7B-Instruct-F16.gguf \
    --port "$PORT" \
    -ngl 99 \
    -c 16384 \
    -t 8 \
    &
LLAMA_PID=$!

# Wait until the server accepts connections (port opens only after model load).
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

# Build just this one seed
python build_lightrag_kg_index_qwenf16_multi.py --size "$SIZE" --version "$VERSION"
EXIT_CODE=$?

exit $EXIT_CODE

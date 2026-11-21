#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/retrieval_%x_%j.out
#SBATCH --job-name=retrieval

# Retrieval-quality metrics (Hit@k / MRR of the gold context among retrieved chunks).
# Only needs the GENERATOR server (hybrid keyword extraction) + the Qwen3 embedder.
# Usage (per-version array):
#   sbatch --array=0-4 --job-name=retr_f16_200 --export=ALL,SIZE=200,IDX_MODEL=f16 slurm_retrieval.sh
#   sbatch --array=0-4 --job-name=retr_q4_200  --export=ALL,SIZE=200,IDX_MODEL=q4  slurm_retrieval.sh

VERSION="${VERSION:-$SLURM_ARRAY_TASK_ID}"
if [ -z "$SIZE" ] || [ -z "$IDX_MODEL" ] || [ -z "$VERSION" ]; then
    echo "ERROR: set SIZE, IDX_MODEL (f16|q4), and a seed (--array=0-4 or VERSION=N)"
    exit 1
fi

source ~/venvs/rag-py311/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load CUDA/12.4.0
mkdir -p logs

if [ "$IDX_MODEL" = "f16" ]; then
    MODEL=~/models/Qwen2.5-7B-Instruct-F16.gguf
else
    MODEL=~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
fi

PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
export GEN_BASE_URL="http://127.0.0.1:${PORT}/v1"
trap 'kill $GEN_PID 2>/dev/null' EXIT

./llama.cpp/build/bin/llama-server -m "$MODEL" --port "$PORT" -ngl 99 -c 16384 -t 8 &
GEN_PID=$!

echo "Waiting for llama-server on :$PORT ..."
for attempt in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; echo "ready"; break; fi
    [ "$attempt" -eq 180 ] && { echo "ERROR: server not ready"; exit 1; }
    sleep 5
done

python retrieval_eval.py --size "$SIZE" --index-model "$IDX_MODEL" --version "$VERSION"
exit $?

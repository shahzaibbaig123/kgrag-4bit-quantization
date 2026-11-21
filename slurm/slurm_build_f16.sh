#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/build_f16_%x_%j.out
#SBATCH --job-name=build_f16

# Usage:
#   sbatch --export=ALL,SIZE=500 slurm_build_f16.sh
#   sbatch --export=ALL,SIZE=2000 slurm_build_f16.sh

if [ -z "$SIZE" ]; then
    echo "ERROR: SIZE not set. Submit with --export=ALL,SIZE=<200|500|2000|4000|8000>"
    exit 1
fi

source ~/venvs/rag-py311/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load CUDA/12.4.0

mkdir -p logs

# Start llama.cpp server (F16 model) on port 8001
./llama.cpp/build/bin/llama-server \
    -m ~/models/Qwen2.5-7B-Instruct-F16.gguf \
    --port 8001 \
    -ngl 99 \
    -c 16384 \
    -t 8 \
    &
LLAMA_PID=$!

# Wait until the server accepts connections. llama-server opens the port
# only after the model finishes loading, so a successful TCP connect means
# it is ready. Bounded (15 min) so a failed load can't hang the allocation.
echo "Waiting for llama-server on :8001 ..."
for attempt in $(seq 1 180); do
    if (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; then
        exec 3>&-
        echo "llama-server :8001 ready."
        break
    fi
    if [ "$attempt" -eq 180 ]; then
        echo "ERROR: llama-server :8001 not ready after 15 min; aborting."
        exit 1
    fi
    sleep 5
done

# Run index building for all versions of this size
python build_lightrag_kg_index_qwenf16_multi.py --size "$SIZE"
EXIT_CODE=$?

# Shut down server
kill "$LLAMA_PID"

exit $EXIT_CODE

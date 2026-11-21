#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=08:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval_Q4_%x_%j.out
#SBATCH --job-name=eval_Q4

# A40 2-GPU eval: judge on GPU 0, generator on GPU 1 (--gpus=2), since the 20B
# judge won't share one 48GB A40 with the generator. For A100 later, one GPU
# fits both (80GB) -> use rag-a100 venv + build_a100.

# Usage (name the job eval_Q4_<size> so timing CSV captures the size):
#   sbatch --job-name=eval_Q4_500  --export=ALL,SIZE=500  slurm_eval_q4.sh
#   sbatch --job-name=eval_Q4_2000 --export=ALL,SIZE=2000 slurm_eval_q4.sh
#
# Runs eval_squad_multi_q4.py --size $SIZE (loops v0-v4).
# Needs TWO llama-servers: generator (Qwen Q4, :8000) + judge (gpt-oss-20b F16_2, :8004).

if [ -z "$SIZE" ]; then
    echo "ERROR: SIZE not set. Submit with --export=ALL,SIZE=<200|500|2000|4000|8000>"
    exit 1
fi

source ~/venvs/rag-py311/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load CUDA/12.4.0

mkdir -p logs

# Kill both servers on any exit (success, failure, or scancel)
trap 'kill $JUDGE_PID $GEN_PID 2>/dev/null' EXIT

# Judge server: gpt-oss-20b F16_2 on port 8004, pinned to GPU 0
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
    -m ~/models/gpt-oss-20b-F16_2.gguf \
    -ngl 99 \
    -c 12288 \
    --host 127.0.0.1 \
    --port 8004 \
    &
JUDGE_PID=$!

# Generator server: Qwen2.5-7B Q4 on port 8000, pinned to GPU 1
CUDA_VISIBLE_DEVICES=1 ./llama.cpp/build/bin/llama-server \
    -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 16384 \
    -t 16 \
    --host 127.0.0.1 \
    --port 8000 \
    &
GEN_PID=$!

# Wait until both servers accept connections. llama-server opens the port
# only after the model finishes loading, so a successful TCP connect means
# ready. Bounded (20 min each) so a failed load can't hang the allocation.
for PORT in 8004 8000; do
    echo "Waiting for llama-server on :$PORT ..."
    for attempt in $(seq 1 240); do
        if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
            exec 3>&-
            echo "llama-server :$PORT ready."
            break
        fi
        if [ "$attempt" -eq 240 ]; then
            echo "ERROR: llama-server :$PORT not ready after 20 min; aborting."
            exit 1
        fi
        sleep 5
    done
done

# Eval: one seed if submitted as an array (--array=0-4), else all 5.
VERSION_ARG=""
[ -n "$SLURM_ARRAY_TASK_ID" ] && VERSION_ARG="--version $SLURM_ARRAY_TASK_ID"
python eval_squad_multi_q4.py --size "$SIZE" $VERSION_ARG
EXIT_CODE=$?

exit $EXIT_CODE

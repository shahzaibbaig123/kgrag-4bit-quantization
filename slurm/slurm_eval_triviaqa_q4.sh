#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/evaltrvi_Q4_%x_%j.out
#SBATCH --job-name=evaltrvi_Q4

# TriviaQA correctness eval (Q4), all sizes 25/50/75/100 in one job.
# Needs: generator (Qwen Q4, :8000) + judge (gpt-oss-20b F16_2, :8004).
#   sbatch slurm_eval_triviaqa_q4.sh

module use /software/easybuild-AMD_A100/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source ~/venvs/rag-a100/bin/activate
mkdir -p logs

trap 'kill $JUDGE_PID $GEN_PID 2>/dev/null' EXIT

./llama.cpp/build_a100/bin/llama-server -m ~/models/gpt-oss-20b-F16_2.gguf \
    -ngl 99 -c 12288 --host 127.0.0.1 --port 8004 &
JUDGE_PID=$!

./llama.cpp/build_a100/bin/llama-server -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 99 -c 16384 -t 16 --host 127.0.0.1 --port 8000 &
GEN_PID=$!

for PORT in 8004 8000; do
    echo "Waiting for llama-server on :$PORT ..."
    for attempt in $(seq 1 240); do
        if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; echo "llama-server :$PORT ready."; break; fi
        [ "$attempt" -eq 240 ] && { echo "ERROR: llama-server :$PORT not ready"; exit 1; }
        sleep 5
    done
done

python eval_triviaqa_q4.py
exit $?

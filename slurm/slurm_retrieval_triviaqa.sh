#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/retrtrvi_%x_%j.out
#SBATCH --job-name=retrtrvi

# TriviaQA retrieval eval (Hit@k / MRR), all sizes 25/50/75/100 in one job.
# Needs only the generator (hybrid query keyword extraction). IDX_MODEL picks it.
#   sbatch --job-name=retrtrvi_f16 --export=ALL,IDX_MODEL=f16 slurm_retrieval_triviaqa.sh
#   sbatch --job-name=retrtrvi_q4  --export=ALL,IDX_MODEL=q4  slurm_retrieval_triviaqa.sh

if [ -z "$IDX_MODEL" ]; then echo "ERROR: IDX_MODEL not set (f16|q4)"; exit 1; fi

module use /software/easybuild-AMD_A100/modules/all
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source ~/venvs/rag-a100/bin/activate
mkdir -p logs

if [ "$IDX_MODEL" = "f16" ]; then MODEL=~/models/Qwen2.5-7B-Instruct-F16.gguf; PORT=8001;
else MODEL=~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf; PORT=8000; fi

trap 'kill $GEN_PID 2>/dev/null' EXIT
./llama.cpp/build_a100/bin/llama-server -m "$MODEL" -ngl 99 -c 16384 -t 16 --host 127.0.0.1 --port "$PORT" &
GEN_PID=$!

echo "Waiting for llama-server on :$PORT ..."
for attempt in $(seq 1 240); do
    if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then exec 3>&-; echo "ready"; break; fi
    [ "$attempt" -eq 240 ] && { echo "ERROR: server not ready"; exit 1; }
    sleep 5
done

python retrieval_eval_triviaqa.py --index-model "$IDX_MODEL"
exit $?

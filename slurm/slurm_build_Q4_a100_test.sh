#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_a100_%x_%j.out
#SBATCH --job-name=test_a100

# Throwaway experiment: does the rag-py311 venv (built on the Intel A40 stack)
# run on an A100 node? Imports the compiled libs that crashed the build.
# Expected: "Illegal instruction (core dumped)" -> confirms the venv/CPU mismatch.

source ~/venvs/rag-py311/bin/activate
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

echo "=== Node: $(hostname) ==="
echo "=== CPU AVX flags ==="
lscpu | grep -oiE "avx512[a-z0-9_]*|avx2|avx" | sort -u | tr '\n' ' '; echo
echo "=== which python ==="
which python
echo "=== import test (SIGILL would happen here) ==="
python -c "import numpy; print('numpy', numpy.__version__); import torch; print('torch', torch.__version__); import sentence_transformers; print('sentence_transformers OK'); import lightrag; print('lightrag OK'); print('>>> ALL IMPORTS OK <<<')"
echo "python exit code: $?"

#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --time=00:05:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/a100_venvtest_%j.out
#SBATCH --job-name=a100_venvtest

# Does the new AMD-built rag-a100 venv run on an A100 (AMD) node?
module use /software/easybuild-AMD_A100/modules/all
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/rag-a100/bin/activate

echo "=== Node: $(hostname) ==="
echo "=== CPU AVX flags ==="
lscpu | grep -oiE "avx512[a-z0-9_]*|avx2|avx" | sort -u | tr '\n' ' '; echo
echo "=== which python ==="
which python
echo "=== build-dep import test (SIGILL would happen here) ==="
python -c "import numpy; print('numpy', numpy.__version__); import torch; print('torch', torch.__version__); import sentence_transformers; print('sentence_transformers OK'); from lightrag import LightRAG; print('lightrag OK'); print('>>> A100 BUILD IMPORTS OK <<<')"
echo "python exit code: $?"

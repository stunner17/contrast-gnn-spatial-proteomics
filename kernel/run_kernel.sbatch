#!/usr/bin/env bash
#SBATCH --job-name=cgnn_kernel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
# Fused Triton contrast kernel: correctness gate, operator benchmark (PyG vs torch.compile vs Triton),
# end-to-end training step. A few minutes. Submit once per GPU type from the repo root:
#   sbatch --partition=<blackwell-partition> kernel/run_kernel.sbatch
#   sbatch --partition=<h200-partition>      kernel/run_kernel.sbatch
cd "${SLURM_SUBMIT_DIR:-.}"
source gnn/env_setup.sh
set -eo pipefail
python -c "import triton" 2>/dev/null || pip install --quiet triton
OUT=results/kernel
mkdir -p $OUT
python kernel/bench_kernel.py --out $OUT

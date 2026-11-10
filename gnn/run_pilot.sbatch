#!/usr/bin/env bash
#SBATCH --job-name=cgnn_pilot
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Pilot: 4 processes share one GPU, each trains one model on DSS fold 0, seed 0.
# Check ms/epoch, peak GPU memory and GPU utilisation before the full run.
# Submit from the repo root: sbatch gnn/run_pilot.sbatch  (add --partition=... for your cluster)
cd "${SLURM_SUBMIT_DIR:-.}"
source gnn/env_setup.sh
set -eo pipefail
OUT=results/02_gnn_pilot
mkdir -p $OUT
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 10 > $OUT/gpu_usage.csv &
SMI=$!
MODELS=(GCN PNA GAT Contrast-GNN+SDpool)
for i in 0 1 2 3; do
  OMP_NUM_THREADS=4 python gnn/train_gnn.py --out $OUT --endpoints DSS --models ${MODELS[$i]} \
      --folds 0 --seeds 0 --max-epochs 300 > $OUT/proc_$i.txt 2>&1 &
done
wait $(jobs -p | grep -v $SMI) || true
kill $SMI || true
cat $OUT/proc_*.txt
awk -F', ' 'NR>1 {gsub(" %","",$3); s+=$3; n++} END {if(n) printf "GPU mean %.1f%% over %d samples\n", s/n, n}' $OUT/gpu_usage.csv

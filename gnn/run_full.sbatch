#!/usr/bin/env bash
#SBATCH --job-name=cgnn_full
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# Full run: 2 endpoints x 20 models x 5 folds x 5 seeds = 1000 tasks, NPROC processes on one GPU.
# Resumable: finished tasks are skipped and each process stops cleanly at 7.4 h; resubmit to continue.
# Reported run: NPROC=8 on a 96 GB GPU; one process ran out of memory, so its 78 tasks were rerun with NPROC=1.
# Submit from the repo root: NPROC=4 sbatch --export=ALL gnn/run_full.sbatch
cd "${SLURM_SUBMIT_DIR:-.}"
source gnn/env_setup.sh
set -eo pipefail
NPROC=${NPROC:-4}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
OUT=results/02_gnn
mkdir -p $OUT
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > $OUT/gpu_usage_${SLURM_JOB_ID}.csv &
SMI=$!
for ((i=0; i<NPROC; i++)); do
  OMP_NUM_THREADS=2 python gnn/train_gnn.py --out $OUT --slice $i/$NPROC > $OUT/proc_${SLURM_JOB_ID}_$i.txt 2>&1 &
done
wait $(jobs -p | grep -v $SMI) || true
kill $SMI || true
echo "finished task files: $(ls $OUT/preds | wc -l) / 1000"
awk -F', ' 'NR>1 {gsub(" %","",$3); s+=$3; n++} END {if(n) printf "GPU mean %.1f%% over %d samples\n", s/n, n}' $OUT/gpu_usage_${SLURM_JOB_ID}.csv

#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=hw3-%A-%a.out

set -euo pipefail

# The submitted directory is the configured repository root with its active Python environment.
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# Each array row owns one GPU and one condition/seed output, with one numerical thread.
conditions=(attention attention attention no_attention no_attention no_attention)
seeds=(42 43 44 42 43 44)
row=${SLURM_ARRAY_TASK_ID:?}
condition=${conditions[$row]}
seed=${seeds[$row]}
folder=research/hw3-three-path-2026-11-09
python "$folder/train.py" --condition "$condition" --seed "$seed" \
  --device cuda --threads 1 --output "$folder/results/$condition/seed$seed"

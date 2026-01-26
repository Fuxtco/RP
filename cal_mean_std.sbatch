#!/bin/bash
#SBATCH --job-name=meanstd_83983
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=06:00:00
#SBATCH --output=/home/xufu518g/SwAV/swav/logs/%x/%j/%j.out
#SBATCH --error=/home/xufu518g/SwAV/swav/logs/%x/%j/%j.err

set -e
set -o pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"

source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
mkdir -p /home/xufu518g/SwAV/swav/logs/$SLURM_JOB_NAME

DATA_ROOT="/projects/p_rep_learn_3/datasets"
MAX_IMAGES=83983

python cal_mean_std.py \
  --data_root "$DATA_ROOT" \
  --max_images -1 \
  --fake_rgb_atol 1e-4

echo "Job finished on $(date)"

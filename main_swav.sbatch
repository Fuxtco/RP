#!/bin/bash
#SBATCH --job-name=adjust_mean_std_only_e150
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=/home/xufu518g/SwAV/swav/logs/train_SwAV/%x/%j/%j.out
#SBATCH --error=/home/xufu518g/SwAV/swav/logs/train_SwAV/%x/%j/%j.err

set -e
set -o pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


cd $SLURM_SUBMIT_DIR
mkdir -p /home/xufu518g/SwAV/swav/logs/train_SwAV/$SLURM_JOB_NAME/$SLURM_JOB_ID
mkdir -p ./output_80k/ddp4/adjust_mean_std_only_e200


torchrun \
  --nproc_per_node=4 \
  main_swav.py \
  --data_path "/projects/p_rep_learn_3/datasets" \
  --nmb_crops 2 4 \
  --size_crops 224 96 \
  --min_scale_crops 0.3 0.10 \
  --max_scale_crops 1.0 0.3 \
  --temperature 0.2 \
  --sinkhorn_iterations 2 \
  --feat_dim 128 \
  --nmb_prototypes 100 \
  --queue_length 4096 \
  --epoch_queue_starts 30 \
  --epochs 200 \
  --checkpoint_freq 50 \
  --batch_size 64 \
  --base_lr 0.025 \
  --final_lr 2e-4 \
  --warmup_epochs 10 \
  --freeze_prototypes_niters 8000 \
  --wd 1e-4 \
  --workers 3 \
  --use_fp16 true \
  --dump_path "./output_80k/ddp4/adjust_mean_std_only_e200"

echo "Job finished on $(date)"

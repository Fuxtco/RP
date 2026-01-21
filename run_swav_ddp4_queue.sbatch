#!/bin/bash
#SBATCH --job-name=fp16_safe_scaleSmall_lrUp_e100
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --output=/home/xufu518g/SwAV/swav/logs/%x/%j/%j.out
#SBATCH --error=/home/xufu518g/SwAV/swav/logs/%x/%j/%j.err

# =========================
# 基础环境设置
# =========================

set -e
set -o pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# 建议明确加载你用过的环境
source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

# 防止 OMP / MKL 抢 CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL 稳定性（HPC 常用）
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# =========================
# 路径准备
# =========================

cd $SLURM_SUBMIT_DIR
mkdir -p /home/xufu518g/SwAV/swav/logs/$SLURM_JOB_NAME
mkdir -p ./output_80k/ddp4/fp16_safe_scaleSmall_lrUp_e100

# =========================
# 启动 SwAV（DDP）
# =========================

torchrun \
  --nproc_per_node=4 \
  main_swav.py \
  --data_path "/projects/p_rep_learn_3/datasets" \
  --nmb_crops 2 4 \
  --size_crops 224 96 \
  --min_scale_crops 0.3 0.05 \
  --max_scale_crops 1.0 0.2 \
  --temperature 0.2 \
  --sinkhorn_iterations 2 \
  --feat_dim 128 \
  --nmb_prototypes 100 \
  --queue_length 4096 \
  --epoch_queue_starts 10 \
  --epochs 100 \
  --checkpoint_freq 5 \
  --batch_size 64 \
  --base_lr 0.025 \
  --final_lr 2e-4 \
  --warmup_epochs 10 \
  --freeze_prototypes_niters 5000 \
  --wd 1e-4 \
  --workers 3 \
  --use_fp16 true \
  --dump_path "./output_80k/ddp4/fp16_safe_scaleSmall_lrUp_e100"

echo "Job finished on $(date)"

#!/bin/bash
#SBATCH --job-name=eval_linear_fp16_safe_scaleSmall_lrUp
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
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

source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

# 防止 OMP / MKL 抢 CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# NCCL 稳定性（可留可不留，单卡影响不大）
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# 避免端口冲突（torchrun 单卡也会用到）
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12000 + RANDOM % 20000))

# =========================
# 路径准备
# =========================
cd $SLURM_SUBMIT_DIR

mkdir -p /home/xufu518g/SwAV/swav/logs/$SLURM_JOB_NAME

# 你的数据：必须包含 train/ 和 val/ 两个文件夹（ImageFolder 结构）
DATA_PATH="./datasets/down"   # <<< 按你真实路径改
CKPT="./output_80k/ddp4/fp16_safe_scaleSmall_lrUp/checkpoint.pth.tar"
OUT_DIR="./eval/test_100/fp16_safe_scaleSmall_lrUp"

mkdir -p "${OUT_DIR}"

# =========================
# 启动 Linear Eval（DDP 1GPU）
# =========================
torchrun --nproc_per_node=2 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  eval_linear.py \
    --data_path "${DATA_PATH}" \
    --pretrained "${CKPT}" \
    --arch resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --workers 2 \
    --dump_path "${OUT_DIR}" \
    --dist_url env://

echo "Job finished on $(date)"

#!/bin/bash
#SBATCH --job-name=pairwise_absdiff
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=/home/xufu518g/SwAV/swav/logs/pairwise/%x/%j/%j.out
#SBATCH --error=/home/xufu518g/SwAV/swav/logs/pairwise/%x/%j/%j.err

set -e
set -o pipefail

source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

DATA=/home/xufu518g/SwAV/swav/datasets/down
CKPT=/home/xufu518g/SwAV/swav/output_80k/ddp4/adjust_mean_std_only/checkpoint.pth.tar
OUT=./eval_pairwise/final/train100_absdiff

mkdir -p "$OUT"

# 1) Train head on train_100
torchrun --nproc_per_node=1 train_linear_sup.py \
  --data_path "$DATA" \
  --train_split train_100 \
  --ckpt "$CKPT" \
  --dump_path "$OUT" \
  --fusion absdiff \
  --epochs 50 \
  --batch_size 256 \
  --workers 8 \
  --use_proj true

# 2) Eval on test_100
torchrun --nproc_per_node=1 eval_linear_sup.py \
  --data_path "$DATA" \
  --split test_100 \
  --ckpt "$CKPT" \
  --head_path "$OUT/pair_head.pth.tar" \
  --dump_path "$OUT/test_100" \
  --fusion absdiff \
  --workers 8 \
  --batch_size 256

# 3) Eval on test_30
torchrun --nproc_per_node=1 eval_linear_sup.py \
  --data_path "$DATA" \
  --split test_30 \
  --ckpt "$CKPT" \
  --head_path "$OUT/pair_head.pth.tar" \
  --dump_path "$OUT/test_30" \
  --fusion absdiff \
  --workers 8 \
  --batch_size 256

# 4) Eval on test_130
torchrun --nproc_per_node=1 eval_linear_sup.py \
  --data_path "$DATA" \
  --split test_130 \
  --ckpt "$CKPT" \
  --head_path "$OUT/pair_head.pth.tar" \
  --dump_path "$OUT/test_130" \
  --fusion absdiff \
  --workers 8 \
  --batch_size 256

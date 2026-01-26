#!/bin/bash
#SBATCH --job-name=cosine_train_4splits_swav_e200_acc
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=/home/xufu518g/SwAV/swav/logs/final/%x/%j/%j.out
#SBATCH --error=/home/xufu518g/SwAV/swav/logs/final/%x/%j/%j.err

set -e
set -o pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate /home/xufu518g/conda_envs/swav_env

# prevent OMP/MKL from hogging CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# =========================
# Paths / Config
# =========================
DATA_ROOT="/home/xufu518g/SwAV/swav/datasets/down"
CKPT="/home/xufu518g/SwAV/swav/output_80k/ddp4/adjust_mean_std_only/checkpoint.pth.tar"

TRAIN_SPLIT="train_100"
TEST100_SPLIT="test_100"
TEST30_SPLIT="test_30"
TEST130_SPLIT="test_130"
UNLAB_SPLIT="Unlabeled"

# evaluation script filename
SCRIPT="eval_cor_cluster.py" 

# Output root (one dir per split)
BASE_OUT="./eval_cluster/final/swav_e200_m_from_${TRAIN_SPLIT}"
OUT_TRAIN="${BASE_OUT}/${TRAIN_SPLIT}"
OUT_TEST100="${BASE_OUT}/${TEST100_SPLIT}"
OUT_TEST30="${BASE_OUT}/${TEST30_SPLIT}"
OUT_TEST130="${BASE_OUT}/${TEST130_SPLIT}"
OUT_UNLAB="${BASE_OUT}/${UNLAB_SPLIT}"

mkdir -p "$OUT_TRAIN" "$OUT_TEST100" "$OUT_TEST30" "$OUT_TEST130" "$OUT_UNLAB"

# Common eval params
BATCH_SIZE=64
WORKERS=4
IMG_SIZE=224
KNN=30
SEED=0

# m sweep config (only used on train_100 when selecting m)
M_INIT=0.55
M_MIN=0.10
M_MAX=0.90
M_STEPS=81

# =========================
# 1) Select m on Train-100 (grid sweep + pick best)
# =========================
python "$SCRIPT" \
  --data_path "$DATA_ROOT" \
  --split "$TRAIN_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --img_size "$IMG_SIZE" \
  --knn "$KNN" \
  --seed "$SEED" \
  --pipeline cosine \
  --m "$M_INIT" \
  --m_min "$M_MIN" \
  --m_max "$M_MAX" \
  --m_steps "$M_STEPS" \
  --select_m_on_train true \
  --m_select_metric ACC \
  --save_m_sweep true \
  --dump_path "$OUT_TRAIN" \
  --solver ilp

# Read selected m from metrics json
METRICS_JSON="${OUT_TRAIN}/metrics_${TRAIN_SPLIT}.json"
if [ ! -f "$METRICS_JSON" ]; then
  echo "[ERROR] Cannot find metrics json: $METRICS_JSON"
  exit 1
fi

MARGIN=$(python - "$METRICS_JSON" << 'PY'
import json, sys
p = sys.argv[1]
with open(p, "r") as f:
    m = json.load(f)
mv = m.get("m", None)
if mv is None:
    raise SystemExit("m not found in metrics json")
print(float(mv))
PY
)

echo "[INFO] Selected m on ${TRAIN_SPLIT}: ${MARGIN}"

# =========================
# 2) Evaluate on test_100 with fixed m (NO sweep, NO selection)
# =========================
python "$SCRIPT" \
  --data_path "$DATA_ROOT" \
  --split "$TEST100_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --img_size "$IMG_SIZE" \
  --knn "$KNN" \
  --seed "$SEED" \
  --pipeline cosine \
  --m "$MARGIN" \
  --select_m_on_train false \
  --save_m_sweep false \
  --solver ilp \
  --dump_path "$OUT_TEST100" \
  --save_montage true \
  --montage_max 25 \
  --montage_cols 5 \
  --save_two_row true \
  --two_row_cols 20 \
  --two_row_img_size 48 \
  --two_row_pad 2 \
  --two_row_gap 12

# =========================
# 3) Evaluate on test_30 with fixed m
# =========================
python "$SCRIPT" \
  --data_path "$DATA_ROOT" \
  --split "$TEST30_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --img_size "$IMG_SIZE" \
  --knn "$KNN" \
  --seed "$SEED" \
  --pipeline cosine \
  --m "$MARGIN" \
  --select_m_on_train false \
  --save_m_sweep false \
  --solver ilp \
  --dump_path "$OUT_TEST30" \
  --save_montage true \
  --montage_max 25 \
  --montage_cols 5 \
  --save_two_row true \
  --two_row_cols 6 \
  --two_row_img_size 48 \
  --two_row_pad 2 \
  --two_row_gap 12

# =========================
# 4) Evaluate on test_130 with fixed m
# =========================
python "$SCRIPT" \
  --data_path "$DATA_ROOT" \
  --split "$TEST130_SPLIT" \
  --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --img_size "$IMG_SIZE" \
  --knn "$KNN" \
  --seed "$SEED" \
  --pipeline cosine \
  --m "$MARGIN" \
  --select_m_on_train false \
  --save_m_sweep false \
  --solver ilp \
  --dump_path "$OUT_TEST130" \
  --save_montage true \
  --montage_max 25 \
  --montage_cols 5 \
  --save_two_row true \
  --two_row_cols 26 \
  --two_row_img_size 48 \
  --two_row_pad 2 \
  --two_row_gap 12

# =========================
# 5) Qualitative clustering on unlabeled with same fixed m
#    (no selection possible because unlabeled has no y_true)
# =========================
python "$SCRIPT" \
  --data_path "$DATA_ROOT" \
  --split "$UNLAB_SPLIT" \
  --unlabeled true \
  --ckpt "$CKPT" \
  --batch_size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --img_size "$IMG_SIZE" \
  --knn "$KNN" \
  --seed "$SEED" \
  --pipeline cosine \
  --m "$MARGIN" \
  --select_m_on_train false \
  --save_m_sweep false \
  --dump_path "$OUT_UNLAB" \
  --save_montage true \
  --montage_max 25 \
  --montage_cols 5 \
  --save_two_row true \
  --two_row_cols 50 \
  --two_row_img_size 24 \
  --two_row_pad 2 \
  --two_row_gap 12

echo "Job finished on $(date)"

#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/default.yaml}
DATA_CSV=${2:-data/clinical_clean.csv}
IMG_ROOT=${3:-data/images}
CKPT_DIR=${4:-experiments/checkpoints}
NUM_WORKERS=${5:-4}

echo "[*] Run training"
echo "  config: $CONFIG"
echo "  data_csv: $DATA_CSV"
echo "  img_root: $IMG_ROOT"
echo "  ckpt_dir: $CKPT_DIR"

python -m src.trainer_main \
  --config ${CONFIG} \
  --data_csv ${DATA_CSV} \
  --img_root ${IMG_ROOT} \
  --ckpt_dir ${CKPT_DIR} \
  --num_workers ${NUM_WORKERS}

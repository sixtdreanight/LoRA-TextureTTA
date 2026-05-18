#!/bin/bash
# 训练所有 mixup 模式（origin/hardest/random/mean），统一参数 K=3 gamma=1.0 alpha=5.0
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
WEIGHTS_DIR="/home/user1/effort/effort_main/Effort-AIGI-Detection-main/DeepfakeBench/training/weights"
TRAIN_DATASET="FaceForensics++"
TEST_DATASET="Celeb-DF-v2"
LOG_DIR="./zhiyuanyan/logs/benchv2/icml25/release"

GAMMA=1.0
ALPHA=5.0
K=3

mkdir -p "$WEIGHTS_DIR"
cp "$YAML" "${YAML}.bak"

# 设置公共参数
sed -i "s/^use_mixup:.*/use_mixup: true/"       "$YAML"
sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/" "$YAML"
sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/" "$YAML"
sed -i "s/^mixup_k:.*/mixup_k: ${K}/"            "$YAML"

train_mode() {
    local NAME=$1
    local MODE=$2
    local SELECTION=$3

    echo "===== Training: ${NAME} ====="

    sed -i "s/^mixup_mode:.*/mixup_mode: ${MODE}/"           "$YAML"
    sed -i "s/^mixup_selection:.*/mixup_selection: ${SELECTION}/" "$YAML"

    local LOG="train_${NAME}.log"
    python3 ./training/train.py \
        --detector_path "$YAML" \
        --train_dataset "$TRAIN_DATASET" \
        --test_dataset "$TEST_DATASET" \
        > "$LOG" 2>&1

    # ckpt 在 effort_<timestamp>/test/Celeb-DF-v2/ckpt_best.pth 下
    local CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/"${TEST_DATASET}"/ckpt_best.pth 2>/dev/null | head -1)
    if [ -n "${CKPT}" ] && [ -f "${CKPT}" ]; then
        cp "${CKPT}" "${WEIGHTS_DIR}/${NAME}.pth"
        echo "Saved: ${WEIGHTS_DIR}/${NAME}.pth"
    else
        echo "WARNING: checkpoint not found under ${LOG_DIR}/effort_*/test/${TEST_DATASET}/"
    fi
}

train_mode "origin_mixup"  "original"   "random"
train_mode "hardest_mixup" "asymmetric" "hardest"
train_mode "random_mixup"  "asymmetric" "random"
train_mode "mean_mixup"    "asymmetric" "mean"

mv "${YAML}.bak" "$YAML"
echo "All done! Weights in $WEIGHTS_DIR"

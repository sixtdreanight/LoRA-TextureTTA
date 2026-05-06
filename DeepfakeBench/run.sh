#!/bin/bash
# 顺序搜索最优 mixup_k，跑完所有候选后输出汇总并写回 yaml
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
TRAIN_DATASET="FaceForensics++"
TEST_DATASET="Celeb-DF-v2"
K_VALUES=(1 2 4 8 16)

# ── 结果存储 ───────────────────────────────────────────────────────────────
declare -A K_ACC   # K_ACC[k] = best_acc

# ── 工具函数 ───────────────────────────────────────────────────────────────
extract_best_acc() {
    local logfile="$1"
    # 日志格式: "testing-metric, acc: 0.9123"
    grep "testing-metric, acc:" "$logfile" \
        | grep -oP "acc:\s*\K[0-9.]+" \
        | sort -n \
        | tail -1
}

# ── 主循环 ─────────────────────────────────────────────────────────────────
for K in "${K_VALUES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  mixup_k = $K  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # 1. 写入 yaml
    sed -i "s/^mixup_k:.*/mixup_k: ${K}/" "$YAML"
    echo "[yaml] mixup_k set to $K"

    # 2. 训练（阻塞，输出到专属日志）
    LOG="train_k${K}.log"
    python3 ./training/train.py \
        --detector_path "$YAML" \
        --train_dataset "$TRAIN_DATASET" \
        --test_dataset  "$TEST_DATASET" \
        > "$LOG" 2>&1

    # 3. 提取最优 acc
    BEST_ACC=$(extract_best_acc "$LOG" || echo "0")
    if [[ -z "$BEST_ACC" ]]; then
        echo "[warn] 未能从 $LOG 解析 acc，跳过"
        BEST_ACC="0"
    fi

    K_ACC[$K]="$BEST_ACC"
    echo "[result] K=$K  best_acc=$BEST_ACC"
done

# ── 汇总比较 ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  搜索汇总"
echo "============================================================"

BEST_K=""
BEST_VAL="-1"

for K in "${K_VALUES[@]}"; do
    ACC="${K_ACC[$K]}"
    echo "  mixup_k=$K   acc=$ACC"
    if (( $(echo "$ACC > $BEST_VAL" | bc -l) )); then
        BEST_VAL="$ACC"
        BEST_K="$K"
    fi
done

echo ""
echo ">>> 最优 mixup_k = $BEST_K  (acc = $BEST_VAL)"

# ── 将最优 K 写回 yaml ─────────────────────────────────────────────────────
sed -i "s/^mixup_k:.*/mixup_k: ${BEST_K}/" "$YAML"
echo ">>> effort.yaml 已更新: mixup_k=$BEST_K"
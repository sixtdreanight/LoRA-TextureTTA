#!/bin/bash
# Mixup sweep v3 — exhaustive search for optimal K, γ, α
# selection=hardest (fixed after ablation showed random degrades)

DETECTOR_PATH="./training/config/detector/effort.yaml"
BACKUP="${DETECTOR_PATH}.bak"
PYMOD=python3
RESULTS_FILE="sweep_results.tsv"

cp "$DETECTOR_PATH" "$BACKUP"
trap "cp $BACKUP $DETECTOR_PATH; rm -f $BACKUP" EXIT

# All K values to try
K_VALS=(1 2 3 5 10)
# Gamma values: 0.2 (conservative), 0.5, 1.0 (neutral), 3.0, 5.0 (aggressive)
GAMMA_VALS=(0.2 0.5 1.0 3.0 5.0)
# Alpha values: lambda distribution concentration
ALPHA_VALS=(5.0)

# Build combos array
declare -a COMBOS=()
for K in "${K_VALS[@]}"; do
    for G in "${GAMMA_VALS[@]}"; do
        for A in "${ALPHA_VALS[@]}"; do
            COMBOS+=("${K}|${G}|${A}")
        done
    done
done

TOTAL=${#COMBOS[@]}
echo "Total combos: ${TOTAL} (K=${#K_VALS[@]} × γ=${#GAMMA_VALS[@]} × α=${#ALPHA_VALS[@]})"
echo ""

echo -e "K\tgamma\talpha\tACC\tAUC\tEER\tAP\tvideo_auc\tvideo_eer\tvideo_acc" > "$RESULTS_FILE"

N=0
for combo in "${COMBOS[@]}"; do
    N=$((N+1))
    IFS='|' read -r K GAMMA ALPHA <<< "$combo"
    LOGFILE="train_K${K}_g${GAMMA}_a${ALPHA}.log"

    echo "=== [${N}/${TOTAL}] K=${K} gamma=${GAMMA} alpha=${ALPHA} -> ${LOGFILE} ==="

    $PYMOD -c "
import yaml
with open('${DETECTOR_PATH}') as f:
    c = yaml.safe_load(f)
c['use_mixup'] = True
c['mixup_k'] = ${K}
c['mixup_gamma'] = float('${GAMMA}')
c['mixup_alpha'] = float('${ALPHA}')
c['mixup_selection'] = 'hardest'
with open('${DETECTOR_PATH}', 'w') as f:
    yaml.dump(c, f, default_flow_style=False, sort_keys=False)
"

    nohup python3 training/train.py \
        --detector_path "$DETECTOR_PATH" \
        --train_dataset FaceForensics++ \
        --test_dataset Celeb-DF-v2 \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "Launched PID=${PID}"
    wait $PID
    echo "Done: ${LOGFILE}"

    METRICS=$($PYMOD -c "
import re
with open('${LOGFILE}') as f:
    text = f.read()
idx = text.rfind('dataset: avg')
if idx < 0:
    print('?\t?\t?\t?\t?\t?\t?')
else:
    chunk = text[idx:]
    m = re.findall(r'testing-metric, (\w+): ([0-9.]+)', chunk)
    d = {k: v for k, v in m}
    keys = ['acc','auc','eer','ap','video_auc','video_eer','video_acc']
    print('\t'.join([d.get(k, '?') for k in keys]))
" 2>/dev/null)

    echo -e "${K}\t${GAMMA}\t${ALPHA}\t${METRICS}" >> "$RESULTS_FILE"
    echo ""
done

echo ""
echo "==================== SWEEP RESULTS ===================="
echo ""

$PYMOD -c "
lines = open('${RESULTS_FILE}').read().strip().split('\n')
header = lines[0]
data = lines[1:]
rows = []
for line in data:
    parts = line.split('\t')
    try:
        va  = float(parts[7])  # col 7: video_auc
        auc = float(parts[4])  # col 4: auc
        acc = float(parts[3])  # col 3: acc
    except:
        va, auc, acc = -1, -1, -1
    rows.append((va, auc, acc, line))

def print_table(sorted_rows, metric_name, idx):
    print(f'=== Sorted by {metric_name} ===')
    for r, (_, _, _, line) in enumerate(sorted_rows):
        m = '  <<< BEST' if r == 0 else ''
        print(f'{line}{m}')
    if sorted_rows and sorted_rows[0][idx] > 0:
        b = sorted_rows[0][3].split('\t')
        print(f'\n>> BEST {metric_name}: K={b[0]} gamma={b[1]} alpha={b[2]}  ACC={b[3]} AUC={b[4]} video_auc={b[7]}')

print_table(sorted(rows, key=lambda x: x[0], reverse=True), 'video_auc', 0)
print()
print_table(sorted(rows, key=lambda x: x[1], reverse=True), 'AUC', 1)
print()
print_table(sorted(rows, key=lambda x: x[2], reverse=True), 'ACC', 2)
"

echo ""
echo "=== All ${TOTAL} sweeps complete ==="

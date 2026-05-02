#!/bin/bash
# Mixup parameter sweep: K ∈ {2,3}, α ∈ {5.0}, γ ∈ {0.2, 3.0}
# After all runs, prints a sorted comparison table and picks the best.

DETECTOR_PATH="./training/config/detector/effort.yaml"
BACKUP="${DETECTOR_PATH}.bak"
PYMOD=python3
RESULTS_FILE="sweep_results.tsv"

cp "$DETECTOR_PATH" "$BACKUP"
trap "cp $BACKUP $DETECTOR_PATH; rm -f $BACKUP" EXIT

declare -a COMBOS=(
    "1|5.0|0.2"
    "1|5.0|3.0"
    "2|5.0|0.2"
    "2|5.0|3.0"
    "3|5.0|0.2"
    "3|5.0|3.0"
)

echo -e "K\talpha\tgamma\tACC\tAUC\tEER\tAP\tvideo_auc\tvideo_eer\tvideo_acc" > "$RESULTS_FILE"

for combo in "${COMBOS[@]}"; do
    IFS='|' read -r K ALPHA GAMMA <<< "$combo"
    LOGFILE="train_K${K}_a${ALPHA}_g${GAMMA}.log"

    echo "=== Running: K=${K} alpha=${ALPHA} gamma=${GAMMA} -> ${LOGFILE} ==="

    $PYMOD -c "
import yaml
with open('${DETECTOR_PATH}') as f:
    c = yaml.safe_load(f)
c['use_mixup'] = True
c['mixup_k'] = ${K}
c['mixup_alpha'] = float('${ALPHA}')
c['mixup_gamma'] = float('${GAMMA}')
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

    # Extract best metrics from final summary (avg line)
    METRICS=$($PYMOD -c "
import re, sys
with open('${LOGFILE}') as f:
    text = f.read()
# find the last occurrence of the avg block
blocks = re.findall(r'avg \w+: [0-9.]+', text[text.rfind('avg acc'):])
d = {}
for b in blocks:
    k, v = b.split(': ')
    d[k.replace('avg ', '')] = v
# also find per-dataset best metric key=value lines before 'avg'
ds_pattern = re.findall(r'\| (\S+): ([0-9.]+) \|', text[text.rfind('dataset_dict'):])
for k, v in ds_pattern:
    d['video_' + k] if k in ('auc','eer','acc') else d.update({k: v})
# output tab-separated in fixed order
order = ['acc','auc','eer','ap','video_auc','video_eer','video_acc']
print('\t'.join([d.get(k, '?') for k in order]))
" 2>/dev/null)

    if [ -n "$METRICS" ]; then
        echo -e "${K}\t${ALPHA}\t${GAMMA}\t${METRICS}" >> "$RESULTS_FILE"
    else
        echo -e "${K}\t${ALPHA}\t${GAMMA}\t?\t?\t?\t?\t?\t?\t?" >> "$RESULTS_FILE"
    fi
    echo ""
done

echo ""
echo "============= SWEEP RESULTS (sorted by video_auc) ============="
echo ""
# Print header and sorted results
head -1 "$RESULTS_FILE"
# Sort by column 7 (video_auc), skip header, descending.
$PYMOD -c "
lines = open('${RESULTS_FILE}').read().strip().split('\n')
header = lines[0]
data = lines[1:]
parsed = []
for line in data:
    parts = line.split('\t')
    try:
        v = float(parts[6])  # video_auc
    except:
        v = -1
    parsed.append((v, line))
parsed.sort(key=lambda x: x[0], reverse=True)
print(header)
for _, line in parsed:
    print(line)
# best
if parsed and parsed[0][0] > 0:
    best = parsed[0][1].split('\t')
    print()
    print(f'>> BEST: K={best[0]} alpha={best[1]} gamma={best[2]}  ACC={best[3]} AUC={best[4]} video_auc={best[6]}')
"

echo ""
echo "=== All sweeps complete ==="

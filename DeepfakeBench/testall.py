"""
Usage:
    python3 testall.py \
        --detector_path ./training/config/detector/effort.yaml \
        --weights_path  /path/to/ckpt_best.pth \
        --test_datasets Celeb-DF-v1 Celeb-DF-v2 DFDC DFDCP FaceForensics++ UADFV
"""

import subprocess
import sys
import re
import argparse
import numpy as np

TEST_SCRIPT = "training/test.py"

parser = argparse.ArgumentParser()
parser.add_argument("--detector_path", required=True)
parser.add_argument("--weights_path",  required=True)
parser.add_argument("--test_datasets", nargs="+", required=True)
args = parser.parse_args()

METRIC_RE = re.compile(r"^([a-zA-Z_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

def parse_metrics(lines):
    result = {}
    for line in lines:
        m = METRIC_RE.match(line.strip())
        if m:
            result[m.group(1)] = float(m.group(2))
    return result

def run_dataset(dataset):
    cmd = [
        sys.executable, TEST_SCRIPT,
        "--detector_path", args.detector_path,
        "--test_dataset",  dataset,
        "--weights_path",  args.weights_path,
    ]

    print(f"\n{'='*60}")
    print(f"dataset: {dataset}")
    print(f"{'='*60}")
    sys.stdout.flush()

    import os
    # stderr goes straight to terminal → tqdm sees a real TTY → progress bar renders normally
    # stdout is captured for metric parsing
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,   # inherit parent's stderr = real terminal
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    collected = []
    for line in proc.stdout:
        line = line.decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        collected.append(line.rstrip("\n"))
    proc.wait()

    if proc.returncode != 0:
        print(f"[WARNING] test.py exited with code {proc.returncode} for {dataset}")

    return parse_metrics(collected)


# ── Run all datasets ──────────────────────────────────────────────────────────
all_metrics = {}

for ds in args.test_datasets:
    metrics = run_dataset(ds)
    if metrics:
        all_metrics[ds] = metrics
    else:
        print(f"[WARNING] No metrics parsed for {ds} — skipping from average.")

# ── Print averages ────────────────────────────────────────────────────────────
if all_metrics:
    metric_names = list(next(iter(all_metrics.values())).keys())
    avg = {}
    for m in metric_names:
        vals = [all_metrics[ds][m] for ds in all_metrics if m in all_metrics[ds]]
        if vals:
            avg[m] = float(np.mean(vals))

    print(f"\n{'='*60}")
    print(f"dataset: average (over {len(all_metrics)} datasets: {', '.join(all_metrics.keys())})")
    for k, v in avg.items():
        print(f"{k}: {v}")

print("\n===> All Done!")

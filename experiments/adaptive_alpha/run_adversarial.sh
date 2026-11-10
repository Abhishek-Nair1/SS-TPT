#!/usr/bin/env bash
# SS-TPT + Adaptive alpha(x) (Task 2): Adversarial PGD (eps=1/255, 7 steps)
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
ALPHA_MODE=${4:-"variance"}
PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets Caltech101 \
    --eps 1.0 --steps 7 --alpha 0.25 \
    --alpha_mode "${ALPHA_MODE}" --ss_ratio 0.4 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/adaptive_alpha/adversarial \
    --save_csv results/adversarial.csv \
    --method_name "SS-TPT + Adaptive α(x)"


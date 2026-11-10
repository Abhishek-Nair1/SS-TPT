#!/usr/bin/env bash
# SS-TPT + Adaptive alpha(x) (Task 2): Clean (Caltech101)
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
ALPHA_MODE=${4:-"variance"}
PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets Caltech101 \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode "${ALPHA_MODE}" --ss_ratio 0.4 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/adaptive_alpha/clean \
    --save_csv results/clean.csv \
    --method_name "SS-TPT + Adaptive α(x)"


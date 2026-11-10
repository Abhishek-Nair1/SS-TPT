#!/usr/bin/env bash
# SS-TPT Baseline: Clean (Caltech101)
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets Caltech101 \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode fixed --ss_ratio 0.4 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/baseline/clean \
    --save_csv results/clean.csv \
    --method_name "SS-TPT baseline"


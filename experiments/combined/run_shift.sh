#!/usr/bin/env bash
# SS-TPT + Spatial + Adaptive alpha(x) (Person 3 Combined): Distribution Shift (ImageNet-R)
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
ALPHA_MODE=${4:-"variance"}
SPAT_WEIGHT=${5:-0.10}
COMBINE_MODE=${6:-"gated"}
PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets R \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode "${ALPHA_MODE}" --ss_ratio 0.4 \
    --use_spatial --spat_weight "${SPAT_WEIGHT}" --combine_mode "${COMBINE_MODE}" --tiny_k 1 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/combined/shift \
    --save_csv results/shift.csv \
    --method_name "SS-TPT + Spatial + Adaptive α(x)"


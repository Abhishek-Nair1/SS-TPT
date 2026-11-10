#!/usr/bin/env bash
# Quick 30-Second Live Demonstration Script (5 Samples)
# Ideal for professor evaluation and quick sanity checks.
set -e

DATA_DIR=${1:-"datasets"}
GPU_ID=${2:-0}
SAMPLES=${3:-5}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "======================================================================"
echo "           SS-TPT QUICK LIVE DEMONSTRATION (5 SAMPLES)"
echo "======================================================================"
echo "Dataset Root: ${DATA_DIR}"
echo "GPU Device:   ${GPU_ID}"
echo "Samples:      ${SAMPLES}"
echo "Python:       $(which ${PYTHON_BIN})"
echo "======================================================================"

echo ""
echo "--> [1/2] Running Clean Caltech-101 Demo on Combined Method..."
${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets Caltech101 \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode variance --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/demo/clean \
    --method_name "SS-TPT + Spatial + Adaptive α(x) [Demo Clean]"

echo ""
echo "--> [2/2] Running Adversarial PGD Demo on Combined Method..."
${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
    --test_sets Caltech101 \
    --eps 1.0 --steps 7 --alpha 0.25 \
    --alpha_mode variance --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu "${GPU_ID}" --max_samples "${SAMPLES}" \
    --output_dir output_results/demo/adversarial \
    --method_name "SS-TPT + Spatial + Adaptive α(x) [Demo Adv]"

echo ""
echo "======================================================================"
echo "Demo complete! Both Clean and Adversarial TTA runs completed on GPU."
echo "======================================================================"

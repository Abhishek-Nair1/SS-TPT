#!/usr/bin/env bash
set -e
cd /home/dse_413/ADL/Final_Project_Folder
export PYTHONUNBUFFERED=1
PYTHON_BIN=/home/dse_413/envs/ss-tpt/bin/python

SAMPLES=${1:-2000}
GPU=${2:-0}

echo "======================================================================"
echo "Starting Complete Run: Task 1 (Spatial) & Combined on GPU ${GPU}"
echo "Samples: ${SAMPLES}"
echo "======================================================================"

mkdir -p output_results/spatial_adv_rtx3070 output_results/combined_adv_rtx3070 results

# 1. Task 1: Spatial Consistency
echo ""
echo "[1/2] Running Task 1: SS-TPT + Spatial on Adversarial Caltech-101 (${SAMPLES} samples)..."
${PYTHON_BIN} sstpt.py data --test_sets Caltech101     --eps 1.0 --steps 7 --alpha 0.25     --alpha_mode fixed --ss_ratio 0.4     --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1     --max_samples ${SAMPLES} --gpu ${GPU}     -p 25     --output_dir output_results/spatial_adv_rtx3070     --save_csv results/adversarial.csv     --method_name "SS-TPT + Spatial"

# 2. Combined: Spatial + Adaptive Alpha
echo ""
echo "[2/2] Running Combined: SS-TPT + Spatial + Adaptive Alpha on Adversarial Caltech-101 (${SAMPLES} samples)..."
${PYTHON_BIN} sstpt.py data --test_sets Caltech101     --eps 1.0 --steps 7 --alpha 0.25     --alpha_mode variance --ss_ratio 0.4     --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1     --max_samples ${SAMPLES} --gpu ${GPU}     -p 25     --output_dir output_results/combined_adv_rtx3070     --save_csv results/adversarial.csv     --method_name "SS-TPT + Spatial + Adaptive α(x)"

echo ""
echo "======================================================================"
echo "Execution Complete! Both methods evaluated on ${SAMPLES} samples."
echo "======================================================================"

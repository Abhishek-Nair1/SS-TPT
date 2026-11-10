#!/usr/bin/env bash
set -e
cd /home/dse_413/ADL/Final_Project_Folder
export PYTHONUNBUFFERED=1
PYTHON_BIN=/home/dse_413/envs/ss-tpt/bin/python

MONITOR_PID=${1:-3583610}
SAMPLES=${2:-2000}
GPU=${3:-0}

echo "[Queue Manager] Waiting for active Adversarial run (PID ${MONITOR_PID}) to finish..."
while kill -0 "${MONITOR_PID}" 2>/dev/null; do
    sleep 15
done

echo "[Queue Manager] Active Adversarial run finished at $(date)!"
echo "[Queue Manager] Starting Clean and ImageNet-R runs for Task 1 and Combined..."

mkdir -p output_results/spatial/clean output_results/combined/clean output_results/spatial/shift output_results/combined/shift results

# --- 1. Clean Caltech-101 ---
echo ""
echo "======================================================================"
echo "[1/4] Running Task 1: SS-TPT + Spatial on Clean Caltech-101 (${SAMPLES} samples)..."
echo "======================================================================"
${PYTHON_BIN} sstpt.py data \
    --test_sets Caltech101 \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode fixed --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu ${GPU} --max_samples ${SAMPLES} \
    -p 50 \
    --output_dir output_results/spatial/clean \
    --save_csv results/clean.csv \
    --method_name "SS-TPT + Spatial"

echo ""
echo "======================================================================"
echo "[2/4] Running Combined: SS-TPT + Spatial + Adaptive Alpha on Clean Caltech-101 (${SAMPLES} samples)..."
echo "======================================================================"
${PYTHON_BIN} sstpt.py data \
    --test_sets Caltech101 \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode variance --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu ${GPU} --max_samples ${SAMPLES} \
    -p 50 \
    --output_dir output_results/combined/clean \
    --save_csv results/clean.csv \
    --method_name "SS-TPT + Spatial + Adaptive α(x)"

# --- 2. Distribution Shift ImageNet-R ---
echo ""
echo "======================================================================"
echo "[3/4] Running Task 1: SS-TPT + Spatial on ImageNet-R (${SAMPLES} samples)..."
echo "======================================================================"
${PYTHON_BIN} sstpt.py data \
    --test_sets R \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode fixed --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu ${GPU} --max_samples ${SAMPLES} \
    -p 50 \
    --output_dir output_results/spatial/shift \
    --save_csv results/shift.csv \
    --method_name "SS-TPT + Spatial"

echo ""
echo "======================================================================"
echo "[4/4] Running Combined: SS-TPT + Spatial + Adaptive Alpha on ImageNet-R (${SAMPLES} samples)..."
echo "====================================================================== "
${PYTHON_BIN} sstpt.py data \
    --test_sets R \
    --eps 0.0 --steps 0 --alpha 0.0 \
    --alpha_mode variance --ss_ratio 0.4 \
    --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
    --gpu ${GPU} --max_samples ${SAMPLES} \
    -p 50 \
    --output_dir output_results/combined/shift \
    --save_csv results/shift.csv \
    --method_name "SS-TPT + Spatial + Adaptive α(x)"

echo ""
echo "======================================================================"
echo "All Clean and ImageNet-R benchmarks complete at $(date)!"
echo "======================================================================"

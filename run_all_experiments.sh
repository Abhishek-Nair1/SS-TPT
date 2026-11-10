#!/usr/bin/env bash
# Master Benchmark Execution Script for All 4 Principal Methods
set -e

DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "=========================================================="
echo "Starting Full 4-Method Benchmark Suite"
echo "Data Directory: ${DATA_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "Sample Cap: ${SAMPLES} (-1 for full set)"
echo "=========================================================="

export PYTHON_BIN

echo "--> [1/4] Running Baseline SS-TPT..."
bash experiments/baseline/run_all.sh "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}"

echo "--> [2/4] Running Task 1: SS-TPT + Spatial..."
bash experiments/spatial/run_all.sh "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}"

echo "--> [3/4] Running Task 2: SS-TPT + Adaptive alpha(x)..."
bash experiments/adaptive_alpha/run_all.sh "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}" "variance"

echo "--> [4/4] Running Combined: SS-TPT + Spatial + Adaptive alpha(x)..."
bash experiments/combined/run_all.sh "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}" "variance" "0.10" "gated"

echo "=========================================================="
echo "All benchmarks finished! Generating analysis and plots..."
${PYTHON_BIN} analysis/statistical_analysis.py
${PYTHON_BIN} analysis/generate_plots.py
${PYTHON_BIN} analysis/generate_report.py
echo "Full evaluation report ready in analysis/FINAL_REPORT.md"
echo "=========================================================="


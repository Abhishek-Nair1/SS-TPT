#!/bin/bash
set -e
PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
GPU_ID=2
SAMPLES=150

echo "=========================================================="
echo "Running Full Validation Sweep strictly on GPU $GPU_ID (80GB A100)"
echo "Tested configurations: Linear, Exp, Sigmoid (w: 0.03-0.20), Temps: 0.15-0.30"
echo "=========================================================="

echo "[1/3] Running Clean (Caltech101) on GPU $GPU_ID..."
$PYTHON_BIN -u run_validation_sweep.py --domain clean --gpu $GPU_ID --samples $SAMPLES

echo "[2/3] Running Adversarial PGD (Caltech101) on GPU $GPU_ID..."
$PYTHON_BIN -u run_validation_sweep.py --domain adv --gpu $GPU_ID --samples $SAMPLES

echo "[3/3] Running Distribution-Shift (ImageNet-R) on GPU $GPU_ID..."
$PYTHON_BIN -u run_validation_sweep.py --domain r --gpu $GPU_ID --samples $SAMPLES

echo "[ALL DONE] Aggregating Results across all 3 domains..."
$PYTHON_BIN -u aggregate_val_results.py


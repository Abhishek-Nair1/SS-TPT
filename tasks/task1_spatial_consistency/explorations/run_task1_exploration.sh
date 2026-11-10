#!/bin/bash
set -e
PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
GPU_ID=2
SAMPLES=150

echo "=========================================================="
echo "Running Task 1 Comprehensive Exploration on GPU $GPU_ID"
echo "Comparing: Pearson (lin/exp), Spatial-JS, Saliency-Pearson, Native-7x7, Entropy-Modulated"
echo "=========================================================="

echo "[1/3] Running Clean (Caltech101) on GPU $GPU_ID..."
$PYTHON_BIN -u run_task1_exploration.py --domain clean --gpu $GPU_ID --samples $SAMPLES

echo "[2/3] Running Adversarial PGD (Caltech101) on GPU $GPU_ID..."
$PYTHON_BIN -u run_task1_exploration.py --domain adv --gpu $GPU_ID --samples $SAMPLES

echo "[3/3] Running Distribution-Shift (ImageNet-R) on GPU $GPU_ID..."
$PYTHON_BIN -u run_task1_exploration.py --domain r --gpu $GPU_ID --samples $SAMPLES

echo "[ALL DONE] Aggregating Results across all 3 domains..."
$PYTHON_BIN -u aggregate_task1_exploration.py

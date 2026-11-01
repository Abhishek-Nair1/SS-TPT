#!/usr/bin/env bash
# ==============================================================================
# Run Adversarial PGD Evaluation on Caltech-101 using Gated S3-TPT
# Attack: PGD eps=1/255, alpha=0.25/255, steps=7
# ==============================================================================
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
GPU_ID=${1:-0}
SAMPLES=${2:-2000}

echo "======================================================================"
echo "Starting Adversarial PGD Evaluation on GPU ${GPU_ID} (${SAMPLES} samples)..."
echo "======================================================================"

${PYTHON_BIN} evaluate.py     --domain adv     --method gated     --samples ${SAMPLES}     --gpu ${GPU_ID}     --weight 0.10     --k 1

#!/usr/bin/env bash
# ==============================================================================
# Run Clean Evaluation on Caltech-101 using Gated S3-TPT (Task 1 Best Solution)
# ==============================================================================
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
GPU_ID=${1:-0}
SAMPLES=${2:-2000}

echo "======================================================================"
echo "Starting Clean Caltech-101 Evaluation on GPU ${GPU_ID} (${SAMPLES} samples)..."
echo "======================================================================"

${PYTHON_BIN} evaluate.py     --domain clean     --method gated     --samples ${SAMPLES}     --gpu ${GPU_ID}     --weight 0.10     --k 1

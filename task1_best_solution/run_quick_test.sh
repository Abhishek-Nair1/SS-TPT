#!/usr/bin/env bash
# ==============================================================================
# Fast Smoke Test (10 samples per domain) to verify environment & correctness
# Runtime: ~40 seconds on GPU 0
# ==============================================================================
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
GPU_ID=${1:-0}

echo "======================================================================"
echo "Running Fast Verification Smoke Test (10 samples per domain on GPU ${GPU_ID})..."
echo "======================================================================"

${PYTHON_BIN} evaluate.py     --domain all     --method gated     --samples 10     --gpu ${GPU_ID}

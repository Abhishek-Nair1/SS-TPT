#!/usr/bin/env bash
# ==============================================================================
# Fast Smoke Test (10 samples per domain) to verify environment & correctness
# Runtime: ~40 seconds on GPU 0
# ==============================================================================
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

if [ -n "${CONDA_PREFIX}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "/home/dse_413/envs/ss-tpt/bin/python" ]; then
    PYTHON_BIN="/home/dse_413/envs/ss-tpt/bin/python"
elif [ -x "/home/abhishek/workspace/conda/sstpt/bin/python" ]; then
    PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
else
    PYTHON_BIN="$(which python3 || which python)"
fi
GPU_ID=${1:-0}

echo "======================================================================"
echo "Running Fast Verification Smoke Test (10 samples per domain on GPU ${GPU_ID})..."
echo "======================================================================"

${PYTHON_BIN} evaluate.py     --domain all     --method gated     --samples 10     --gpu ${GPU_ID}

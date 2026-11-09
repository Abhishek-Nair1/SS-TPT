#!/usr/bin/env bash
# ==============================================================================
# Run Full Official 3-Domain Benchmark (Clean, Adversarial, Distribution-Shift)
# Optional flag --compare runs side-by-side baseline vs gated S3-TPT comparison
# ==============================================================================
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

if [ -n "${CONDA_PREFIX}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "/home/dse_413/envs/ss-tpt/bin/python" ]; then
    PYTHON_BIN="/home/dse_413/envs/ss-tpt/bin/python"
elif [ -x "/home/abhishek/workspace/conda/sstpt/bin/python" ]; then
    if [ -n "${CONDA_PREFIX}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [ -x "/home/dse_413/envs/ss-tpt/bin/python" ]; then
    PYTHON_BIN="/home/dse_413/envs/ss-tpt/bin/python"
elif [ -x "/home/abhishek/workspace/conda/sstpt/bin/python" ]; then
    PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
else
    PYTHON_BIN="$(which python3 || which python)"
fi
else
    PYTHON_BIN="$(which python3 || which python)"
fi
GPU_ID=${1:-0}
SAMPLES=${2:-2000}
METHOD=${3:-gated}

echo "======================================================================"
echo "Running Full Benchmark across Clean, Adversarial PGD, and ImageNet-R"
echo "Method: ${METHOD} | GPU: ${GPU_ID} | Samples: ${SAMPLES}"
echo "======================================================================"

${PYTHON_BIN} evaluate.py     --domain all     --method ${METHOD}     --samples ${SAMPLES}     --gpu ${GPU_ID}

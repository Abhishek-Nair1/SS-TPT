#!/usr/bin/env bash
# SS-TPT + Spatial + Adaptive alpha(x) (Person 3 Combined): Run All 3 Benchmarks
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}
ALPHA_MODE=${4:-"variance"}
SPAT_WEIGHT=${5:-0.10}
COMBINE_MODE=${6:-"gated"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_clean.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}" "${ALPHA_MODE}" "${SPAT_WEIGHT}" "${COMBINE_MODE}"
bash "${SCRIPT_DIR}/run_adversarial.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}" "${ALPHA_MODE}" "${SPAT_WEIGHT}" "${COMBINE_MODE}"
bash "${SCRIPT_DIR}/run_shift.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}" "${ALPHA_MODE}" "${SPAT_WEIGHT}" "${COMBINE_MODE}"


#!/usr/bin/env bash
# SS-TPT + Spatial (Task 1): Run All 3 Benchmarks
set -e
DATA_DIR=${1:-"data"}
GPU_ID=${2:-0}
SAMPLES=${3:--1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_clean.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}"
bash "${SCRIPT_DIR}/run_adversarial.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}"
bash "${SCRIPT_DIR}/run_shift.sh" "${DATA_DIR}" "${GPU_ID}" "${SAMPLES}"


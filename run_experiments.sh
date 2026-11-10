#!/usr/bin/env bash
# ==============================================================================
# Unified SS-TPT Experiment Runner
# Evaluates any or all 4 principal methods across 3 test regimes:
#   1. Clean (Caltech-101, eps=0.0)
#   2. Adversarial PGD (Caltech-101, eps=1.0, 7 steps)
#   3. Distribution Shift (ImageNet-R, eps=0.0)
#
# Usage:
#   ./run_experiments.sh [METHOD] [REGIME] [SAMPLES] [GPU_ID]
#
# Examples:
#   ./run_experiments.sh all all 2000 0        # Run full master benchmark
#   ./run_experiments.sh combined adv 500 0    # Test combined on adversarial
#   ./run_experiments.sh spatial clean 2000 0  # Test spatial on clean
# ==============================================================================
set -e

METHOD=${1:-"all"}       # Options: all, baseline, spatial, alpha, combined
REGIME=${2:-"all"}       # Options: all, clean, adv, shift
SAMPLES=${3:-2000}       # Default: 2000 samples
GPU_ID=${4:-0}           # Default: GPU 0
PYTHON_BIN=${PYTHON_BIN:-python}

DATA_DIR=${DATA_DIR:-"data"}
OUTPUT_DIR="output_results"
mkdir -p "${OUTPUT_DIR}" results

run_single() {
    local m_name="$1"
    local d_name="$2"
    local eps="$3"
    local steps="$4"
    local alpha_mode="$5"
    local use_spatial="$6"
    local spat_w="$7"
    local csv_name="$8"

    echo ""
    echo "======================================================================"
    echo "Running: ${m_name} on ${d_name} (Samples: ${SAMPLES}, GPU: ${GPU_ID})"
    echo "======================================================================"

    local spat_flag=""
    if [ "${use_spatial}" = "true" ]; then
        spat_flag="--use_spatial --spat_weight ${spat_w} --combine_mode gated --tiny_k 1"
    fi

    ${PYTHON_BIN} sstpt.py "${DATA_DIR}" \
        --test_sets "${d_name}" \
        --eps "${eps}" \
        --steps "${steps}" \
        --alpha 0.25 \
        --alpha_mode "${alpha_mode}" \
        --ss_ratio 0.4 \
        ${spat_flag} \
        --max_samples "${SAMPLES}" \
        --gpu "${GPU_ID}" \
        -p 25 \
        --output_dir "${OUTPUT_DIR}/${m_name// /_}" \
        --save_csv "results/${csv_name}.csv" \
        --method_name "${m_name}"
}

run_regime() {
    local reg="$1"
    local m="$2"

    local d_set="Caltech101"
    local eps="0.0"
    local steps="0"
    local csv="clean"

    if [ "${reg}" = "adv" ]; then
        d_set="Caltech101"
        eps="1.0"
        steps="7"
        csv="adversarial"
    elif [ "${reg}" = "shift" ]; then
        d_set="R"
        eps="0.0"
        steps="0"
        csv="shift"
    fi

    # 1. Baseline
    if [ "${m}" = "all" ] || [ "${m}" = "baseline" ]; then
        run_single "SS-TPT baseline" "${d_set}" "${eps}" "${steps}" "fixed" "false" "0.0" "${csv}"
    fi

    # 2. Spatial (Task 1)
    if [ "${m}" = "all" ] || [ "${m}" = "spatial" ]; then
        run_single "SS-TPT + Spatial" "${d_set}" "${eps}" "${steps}" "fixed" "true" "0.10" "${csv}"
    fi

    # 3. Adaptive Alpha (Task 2)
    if [ "${m}" = "all" ] || [ "${m}" = "alpha" ]; then
        run_single "SS-TPT + Adaptive α(x)" "${d_set}" "${eps}" "${steps}" "variance" "false" "0.0" "${csv}"
    fi

    # 4. Combined
    if [ "${m}" = "all" ] || [ "${m}" = "combined" ]; then
        run_single "SS-TPT + Spatial + Adaptive α(x)" "${d_set}" "${eps}" "${steps}" "variance" "true" "0.10" "${csv}"
    fi
}

echo "======================================================================"
echo "          SS-TPT BENCHMARK EXECUTION HARNESS"
echo "======================================================================"
echo "Method:     ${METHOD}"
echo "Regime:     ${REGIME}"
echo "Samples:    ${SAMPLES}"
echo "GPU Device: ${GPU_ID}"
echo "Python:     $(which ${PYTHON_BIN})"
echo "======================================================================"

if [ "${REGIME}" = "all" ]; then
    run_regime "clean" "${METHOD}"
    run_regime "adv" "${METHOD}"
    run_regime "shift" "${METHOD}"
else
    run_regime "${REGIME}" "${METHOD}"
fi

echo ""
echo "======================================================================"
echo "All requested evaluations completed successfully!"
echo "Results logged to: results/"
echo "======================================================================"

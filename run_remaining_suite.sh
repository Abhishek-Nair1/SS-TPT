#!/usr/bin/env bash
# Runner for remaining 5 benchmarks of SS-TPT Suite across 3 A100 GPUs
set -e

PROJECT_DIR="/home/abhishek/workspace/Final_Project_Folder"
PYTHON_BIN="/home/abhishek/workspace/conda/sstpt/bin/python"
LOG_FILE="${PROJECT_DIR}/master_benchmark_pragyan_2000.log"
SAMPLES=2000

cd "${PROJECT_DIR}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

log "================================================================================"
log "Resuming Remaining 5 Benchmarks in Parallel on 3x NVIDIA A100 GPUs"
log "GPU 0: Combined Clean -> Combined Shift -> Adaptive Alpha Shift"
log "GPU 1: Combined Adversarial PGD"
log "GPU 2: Adaptive Alpha Adversarial PGD"
log "================================================================================"

# --- Worker for GPU 1 (Combined Adv) ---
(
    log "[GPU 1] >>> STARTING: SS-TPT Combined | Adversarial PGD (Caltech101)"
    ${PYTHON_BIN} sstpt.py data --test_sets Caltech101 \
        --eps 1.0 --steps 7 --alpha 0.25 \
        --alpha_mode variance --ss_ratio 0.4 \
        --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
        --gpu 1 --max_samples ${SAMPLES} \
        --output_dir output_results/combined/adversarial \
        --save_csv results/adversarial.csv \
        --method_name "SS-TPT + Spatial + Adaptive α(x)"
    log "[GPU 1] <<< FINISHED: SS-TPT Combined | Adversarial PGD"
) >> "${LOG_FILE}" 2>&1 &
PID_GPU1=$!

# --- Worker for GPU 2 (Adaptive Adv) ---
(
    log "[GPU 2] >>> STARTING: SS-TPT + Adaptive α(x) | Adversarial PGD (Caltech101)"
    ${PYTHON_BIN} sstpt.py data --test_sets Caltech101 \
        --eps 1.0 --steps 7 --alpha 0.25 \
        --alpha_mode variance --ss_ratio 0.4 \
        --gpu 2 --max_samples ${SAMPLES} \
        --output_dir output_results/adaptive_alpha/adversarial \
        --save_csv results/adversarial.csv \
        --method_name "SS-TPT + Adaptive α(x)"
    log "[GPU 2] <<< FINISHED: SS-TPT + Adaptive α(x) | Adversarial PGD"
) >> "${LOG_FILE}" 2>&1 &
PID_GPU2=$!

# --- Worker for GPU 0 (Combined Clean -> Combined Shift -> Adaptive Shift) ---
(
    log "[GPU 0] >>> STARTING: SS-TPT Combined | Clean (Caltech101)"
    ${PYTHON_BIN} sstpt.py data --test_sets Caltech101 \
        --eps 0.0 --steps 0 --alpha 0.0 \
        --alpha_mode variance --ss_ratio 0.4 \
        --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
        --gpu 0 --max_samples ${SAMPLES} \
        --output_dir output_results/combined/clean \
        --save_csv results/clean.csv \
        --method_name "SS-TPT + Spatial + Adaptive α(x)"
    log "[GPU 0] <<< FINISHED: SS-TPT Combined | Clean"

    log "[GPU 0] >>> STARTING: SS-TPT Combined | Shift (ImageNet-R)"
    ${PYTHON_BIN} sstpt.py data --test_sets R \
        --eps 0.0 --steps 0 --alpha 0.0 \
        --alpha_mode variance --ss_ratio 0.4 \
        --use_spatial --spat_weight 0.10 --combine_mode gated --tiny_k 1 \
        --gpu 0 --max_samples ${SAMPLES} \
        --output_dir output_results/combined/shift \
        --save_csv results/shift.csv \
        --method_name "SS-TPT + Spatial + Adaptive α(x)"
    log "[GPU 0] <<< FINISHED: SS-TPT Combined | Shift"

    log "[GPU 0] >>> STARTING: SS-TPT + Adaptive α(x) | Shift (ImageNet-R)"
    ${PYTHON_BIN} sstpt.py data --test_sets R \
        --eps 0.0 --steps 0 --alpha 0.0 \
        --alpha_mode variance --ss_ratio 0.4 \
        --gpu 0 --max_samples ${SAMPLES} \
        --output_dir output_results/adaptive_alpha/shift \
        --save_csv results/shift.csv \
        --method_name "SS-TPT + Adaptive α(x)"
    log "[GPU 0] <<< FINISHED: SS-TPT + Adaptive α(x) | Shift"
) >> "${LOG_FILE}" 2>&1 &
PID_GPU0=$!

# Wait for all 3 GPU workers
wait ${PID_GPU0}
wait ${PID_GPU1}
wait ${PID_GPU2}

log "================================================================================"
log "ALL 12 BENCHMARKS COMPLETED SUCCESSFULLY!"
log "Generating final statistical analysis, plots, and updated report..."
log "================================================================================"

${PYTHON_BIN} analysis/statistical_analysis.py >> "${LOG_FILE}" 2>&1
${PYTHON_BIN} analysis/generate_plots.py >> "${LOG_FILE}" 2>&1
${PYTHON_BIN} analysis/generate_report.py >> "${LOG_FILE}" 2>&1

log "================================================================================"
log "SUITE COMPLETE! Final Report: ${PROJECT_DIR}/analysis/FINAL_REPORT.md"
log "================================================================================"


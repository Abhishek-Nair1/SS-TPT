#!/usr/bin/env python3
"""Master 3-GPU Parallel Benchmark Runner for SS-TPT Suite on pragyan.

Distributes all 12 benchmarks (4 methods x 3 domains: Clean, Adversarial, Shift)
across 3 NVIDIA A100 GPUs (GPU 0, GPU 1, GPU 2) dynamically via a work queue.
"""

import os
import sys
import time
import subprocess
import threading
import queue
from datetime import datetime

PYTHON_BIN = "/home/abhishek/workspace/conda/sstpt/bin/python"
PROJECT_DIR = "/home/abhishek/workspace/Final_Project_Folder"
DATA_DIR = "data"
SAMPLES = 2000
GPU_IDS = [0, 1, 1, 2, 2]
MASTER_LOG_PATH = os.path.join(PROJECT_DIR, "master_benchmark_2000_verified.log")

log_lock = threading.Lock()

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with log_lock:
        with open(MASTER_LOG_PATH, "a") as f:
            f.write(formatted + "\n")
            f.flush()

JOBS = [
    # 1. Baseline SS-TPT
    {
        "id": "baseline_clean",
        "name": "SS-TPT Baseline | Clean (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/baseline/clean",
            "--save_csv", "results/clean.csv",
            "--method_name", "SS-TPT baseline"
        ]
    },
    {
        "id": "baseline_shift",
        "name": "SS-TPT Baseline | Shift (ImageNet-R)",
        "args": [
            "data", "--test_sets", "R",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/baseline/shift",
            "--save_csv", "results/shift.csv",
            "--method_name", "SS-TPT baseline"
        ]
    },
    {
        "id": "baseline_adv",
        "name": "SS-TPT Baseline | Adversarial PGD (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "1.0", "--steps", "7", "--alpha", "0.25",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/baseline/adversarial",
            "--save_csv", "results/adversarial.csv",
            "--method_name", "SS-TPT baseline"
        ]
    },

    # 2. Task 1: SS-TPT + Spatial
    {
        "id": "spatial_clean",
        "name": "SS-TPT + Spatial | Clean (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/spatial/clean",
            "--save_csv", "results/clean.csv",
            "--method_name", "SS-TPT + Spatial"
        ]
    },
    {
        "id": "spatial_shift",
        "name": "SS-TPT + Spatial | Shift (ImageNet-R)",
        "args": [
            "data", "--test_sets", "R",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/spatial/shift",
            "--save_csv", "results/shift.csv",
            "--method_name", "SS-TPT + Spatial"
        ]
    },
    {
        "id": "spatial_adv",
        "name": "SS-TPT + Spatial | Adversarial PGD (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "1.0", "--steps", "7", "--alpha", "0.25",
            "--alpha_mode", "fixed", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/spatial/adversarial",
            "--save_csv", "results/adversarial.csv",
            "--method_name", "SS-TPT + Spatial"
        ]
    },

    # 3. Task 2: SS-TPT + Adaptive alpha(x)
    {
        "id": "adaptive_clean",
        "name": "SS-TPT + Adaptive α(x) | Clean (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/adaptive_alpha/clean",
            "--save_csv", "results/clean.csv",
            "--method_name", "SS-TPT + Adaptive α(x)"
        ]
    },
    {
        "id": "adaptive_shift",
        "name": "SS-TPT + Adaptive α(x) | Shift (ImageNet-R)",
        "args": [
            "data", "--test_sets", "R",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/adaptive_alpha/shift",
            "--save_csv", "results/shift.csv",
            "--method_name", "SS-TPT + Adaptive α(x)"
        ]
    },
    {
        "id": "adaptive_adv",
        "name": "SS-TPT + Adaptive α(x) | Adversarial PGD (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "1.0", "--steps", "7", "--alpha", "0.25",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/adaptive_alpha/adversarial",
            "--save_csv", "results/adversarial.csv",
            "--method_name", "SS-TPT + Adaptive α(x)"
        ]
    },

    # 4. Combined: SS-TPT + Spatial + Adaptive alpha(x)
    {
        "id": "combined_clean",
        "name": "SS-TPT Combined | Clean (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/combined/clean",
            "--save_csv", "results/clean.csv",
            "--method_name", "SS-TPT + Spatial + Adaptive α(x)"
        ]
    },
    {
        "id": "combined_shift",
        "name": "SS-TPT Combined | Shift (ImageNet-R)",
        "args": [
            "data", "--test_sets", "R",
            "--eps", "0.0", "--steps", "0", "--alpha", "0.0",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/combined/shift",
            "--save_csv", "results/shift.csv",
            "--method_name", "SS-TPT + Spatial + Adaptive α(x)"
        ]
    },
    {
        "id": "combined_adv",
        "name": "SS-TPT Combined | Adversarial PGD (Caltech101)",
        "args": [
            "data", "--test_sets", "Caltech101",
            "--eps", "1.0", "--steps", "7", "--alpha", "0.25",
            "--alpha_mode", "variance", "--ss_ratio", "0.4",
            "--use_spatial", "--spat_weight", "0.10", "--combine_mode", "gated", "--tiny_k", "1",
            "--max_samples", str(SAMPLES),
            "--output_dir", "output_results/combined/adversarial",
            "--save_csv", "results/adversarial.csv",
            "--method_name", "SS-TPT + Spatial + Adaptive α(x)"
        ]
    },
]

job_queue = queue.Queue()
for job in JOBS:
    job_queue.put(job)

completed_jobs = 0
total_jobs = len(JOBS)

def worker(gpu_id):
    global completed_jobs
    while not job_queue.empty():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        job_id = job["id"]
        job_name = job["name"]
        cmd = [PYTHON_BIN, "sstpt.py"] + job["args"] + ["--gpu", str(gpu_id)]

        log(f"[GPU {gpu_id}] >>> STARTING: {job_name}")
        start_t = time.time()

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )

        for line in proc.stdout:
            line_str = line.strip()
            if "iter:" in line_str or "Top-1 Accuracy:" in line_str or "Error" in line_str or "Traceback" in line_str:
                log(f"[GPU {gpu_id} | {job_id}] {line_str}")

        proc.wait()
        elapsed = time.time() - start_t

        with log_lock:
            completed_jobs += 1
            cur_done = completed_jobs

        if proc.returncode == 0:
            log(f"[GPU {gpu_id}] <<< FINISHED: {job_name} ({elapsed:.1f}s) [{cur_done}/{total_jobs} complete]")
        else:
            log(f"[GPU {gpu_id}] !!! FAILED (code {proc.returncode}): {job_name} ({elapsed:.1f}s)")

        job_queue.task_done()

def main():
    total_start = time.time()
    log("================================================================================")
    log(f"Starting 3-GPU Parallel Execution of 4-Method Benchmark Suite on pragyan")
    log(f"GPUs Allocated: {GPU_IDS} (3x NVIDIA A100)")
    log(f"Samples per Benchmark: {SAMPLES}")
    log(f"Total Benchmarks to Run: {len(JOBS)}")
    log("================================================================================")

    threads = []
    for gpu_id in GPU_IDS:
        t = threading.Thread(target=worker, args=(gpu_id,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_time = time.time() - total_start
    log("================================================================================")
    log(f"All 12 benchmarks completed in {total_time:.1f}s ({total_time / 60:.2f} minutes)!")
    log("Generating statistical analysis, plots, and updated final report...")
    log("================================================================================")

    for script in ["statistical_analysis.py", "generate_plots.py", "generate_report.py"]:
        script_path = os.path.join(PROJECT_DIR, "analysis", script)
        log(f"Running: analysis/{script}")
        res = subprocess.run([PYTHON_BIN, script_path], cwd=PROJECT_DIR, capture_output=True, text=True)
        if res.returncode == 0:
            log(f"analysis/{script} completed successfully.")
        else:
            log(f"analysis/{script} error: {res.stderr[:200]}")

    log("================================================================================")
    log("MASTER BENCHMARK RUN FULLY COMPLETE!")
    log(f"Final Report: {os.path.join(PROJECT_DIR, 'analysis', 'FINAL_REPORT.md')}")
    log("================================================================================")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Unified Benchmark Driver for Person 3 Evaluation Harness.

Executes and logs all 4 principal methods across the 3 benchmark regimes:
  1. SS-TPT baseline
  2. SS-TPT + Spatial (Task 1)
  3. SS-TPT + Adaptive alpha(x) (Task 2)
  4. SS-TPT + Spatial + Adaptive alpha(x) (Person 3 Integration)

Outputs standardized CSV records into:
  - results/clean.csv
  - results/adversarial.csv
  - results/shift.csv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

DOMAINS = {
    "clean": {
        "name": "Clean (Caltech101)",
        "dataset": "Caltech101",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
        "csv_file": RESULTS_DIR / "clean.csv",
    },
    "adv": {
        "name": "Adversarial PGD (Caltech101)",
        "dataset": "Caltech101",
        "eps": 1.0,
        "steps": 7,
        "alpha": 0.25,
        "csv_file": RESULTS_DIR / "adversarial.csv",
    },
    "shift": {
        "name": "Distribution Shift (ImageNet-R)",
        "dataset": "R",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
        "csv_file": RESULTS_DIR / "shift.csv",
    },
}

METHODS = {
    "baseline": {
        "name": "SS-TPT baseline",
        "flags": ["--alpha_mode", "fixed", "--ss_ratio", "0.4"],
    },
    "spatial": {
        "name": "SS-TPT + Spatial",
        "flags": [
            "--alpha_mode", "fixed",
            "--ss_ratio", "0.4",
            "--use_spatial",
            "--spat_weight", "0.10",
            "--combine_mode", "gated",
            "--tiny_k", "1",
        ],
    },
    "adaptive_alpha": {
        "name": "SS-TPT + Adaptive α(x)",
        "flags": ["--alpha_mode", "variance", "--ss_ratio", "0.4"],
    },
    "combined": {
        "name": "SS-TPT + Spatial + Adaptive α(x)",
        "flags": [
            "--alpha_mode", "variance",
            "--ss_ratio", "0.4",
            "--use_spatial",
            "--spat_weight", "0.10",
            "--combine_mode", "gated",
            "--tiny_k", "1",
        ],
    },
}


def run_experiment(
    method_key,
    domain_key,
    data_dir,
    python_bin,
    gpu_id=0,
    max_samples=-1,
    seed=0,
    extra_flags=None,
):
    method = METHODS[method_key]
    domain = DOMAINS[domain_key]

    out_dir = REPO_ROOT / "output_results" / method_key / domain_key
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_bin),
        str(REPO_ROOT / "sstpt.py"),
        str(data_dir),
        "--test_sets", domain["dataset"],
        "--eps", str(domain["eps"]),
        "--steps", str(domain["steps"]),
        "--alpha", str(domain["alpha"]),
        "--gpu", str(gpu_id),
        "--seed", str(seed),
        "--max_samples", str(max_samples),
        "--output_dir", str(out_dir),
        "--save_csv", str(domain["csv_file"]),
        "--method_name", method["name"],
    ]
    cmd.extend(method["flags"])
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"\n=======================================================")
    print(f"Running: {method['name']} on {domain['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")

    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"Error: Experiment failed with code {res.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Person 3 Evaluation Benchmark Driver")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "baseline", "spatial", "adaptive_alpha", "combined"],
        help="method to benchmark",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "clean", "adv", "shift"],
        help="test domain",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="path to dataset root",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=sys.executable,
        help="path to python executable",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--samples", type=int, default=-1, help="sample limit (-1 for full set)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    selected_methods = (
        ["baseline", "spatial", "adaptive_alpha", "combined"]
        if args.method == "all"
        else [args.method]
    )
    selected_domains = (
        ["clean", "adv", "shift"] if args.domain == "all" else [args.domain]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for m in selected_methods:
        for d in selected_domains:
            run_experiment(
                method_key=m,
                domain_key=d,
                data_dir=args.data_dir,
                python_bin=args.python_bin,
                gpu_id=args.gpu,
                max_samples=args.samples,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Evaluation & Benchmark Driver for Task 1: Cross-View Spatial Consistency Scoring (Gated S3-TPT).

This script allows running:
  1. Clean Caltech-101 evaluation
  2. Adversarial PGD (eps=1/255, alpha=0.25/255, steps=7) on Caltech-101
  3. Natural Distribution Shift on ImageNet-R
  4. Full Benchmark across all 3 domains
  5. Side-by-side head-to-head comparison between Original SS-TPT and Gated S3-TPT

Example usages:
  # Quick smoke test on 20 samples on Clean
  python evaluate.py --domain clean --samples 20 --gpu 0

  # Evaluate Gated S3-TPT on Adversarial PGD on 100 samples
  python evaluate.py --domain adv --samples 100 --gpu 0

  # Head-to-head comparison on all 3 domains (2000 samples)
  python evaluate.py --domain all --method both --samples 2000 --gpu 0
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
import argparse
import subprocess
import re
import json
import time
from pathlib import Path

# Self-contained paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data" if (SCRIPT_DIR / "data").exists() else REPO_ROOT / "data"
PYTHON_BIN = "/home/abhishek/workspace/conda/sstpt/bin/python"

DOMAINS = {
    "clean": {
        "name": "Clean (Caltech101)",
        "dataset": "Caltech101",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
    },
    "adv": {
        "name": "Adversarial PGD (Caltech101)",
        "dataset": "Caltech101",
        "eps": 1.0,
        "steps": 7,
        "alpha": 0.25,
    },
    "r": {
        "name": "Distribution-Shift (ImageNet-R)",
        "dataset": "R",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
    }
}

METHOD_CONFIGS = {
    "baseline": {
        "name": "Original SS-TPT Baseline",
        "spat_weight": 0.0,
        "combine_mode": "gated",
        "tiny_k": 1,
        "selection_mode": "entropy",
        "spat_method": "attn"
    },
    "gated": {
        "name": "Gated S3-TPT (Task 1 Best Solution)",
        "spat_weight": 0.10,
        "combine_mode": "gated",
        "tiny_k": 1,
        "selection_mode": "entropy",
        "spat_method": "attn"
    },
    "gated_k2": {
        "name": "Gated S3-TPT (K=2 Augmentations)",
        "spat_weight": 0.10,
        "combine_mode": "gated",
        "tiny_k": 2,
        "selection_mode": "entropy",
        "spat_method": "attn"
    }
}


def run_single_eval(domain_key, method_key, max_samples, gpu_id, spat_weight, tiny_k, out_dir):
    dom_cfg = DOMAINS[domain_key]
    m_cfg = METHOD_CONFIGS[method_key]
    
    # Allow overriding spat_weight and tiny_k via CLI
    eff_spat_weight = spat_weight if spat_weight is not None else m_cfg["spat_weight"]
    eff_k = tiny_k if tiny_k is not None else m_cfg["tiny_k"]
    
    print("\n======================================================================")
    print(f"Running: {dom_cfg['name']} | Method: {m_cfg['name']}")
    print(f"Settings: spat_weight={eff_spat_weight}, K={eff_k}, samples={max_samples}, GPU={gpu_id}")
    print("======================================================================")
    
    cmd = [
        PYTHON_BIN, "-u", str(SCRIPT_DIR / "sstpt.py"), str(DATA_DIR),
        "-a", "RN50",
        "-b", "16",
        "--gpu", str(gpu_id),
        "--ctx_init", "a_photo_of_a",
        "-p", "25",
        "--eps", str(dom_cfg["eps"]),
        "--steps", str(dom_cfg["steps"]),
        "--alpha", str(dom_cfg["alpha"]),
        "--test_sets", dom_cfg["dataset"],
        "--output_dir", str(out_dir / "ckps"),
        "--max_samples", str(max_samples),
        "--spat_weight", str(eff_spat_weight),
        "--combine_mode", m_cfg["combine_mode"],
        "--spat_method", m_cfg["spat_method"],
        "--selection_mode", m_cfg["selection_mode"],
        "--tiny_k", str(eff_k)
    ]
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    full_output = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        full_output.append(line)
    proc.wait()
    total_elapsed = time.time() - start_time
    output_str = "".join(full_output)
    
    # Parse final accuracy
    acc_match = re.search(r"\*\s+SSTPTAcc@1\s+([\d\.]+)", output_str)
    if acc_match:
        accuracy = float(acc_match.group(1))
    else:
        iter_matches = re.findall(r"sstpt_(?:acc|adv)1=([\d\.]+)", output_str)
        accuracy = float(iter_matches[-1]) if iter_matches else 0.0

    return {
        "domain": dom_cfg["name"],
        "domain_key": domain_key,
        "method": m_cfg["name"],
        "method_key": method_key,
        "spat_weight": eff_spat_weight,
        "tiny_k": eff_k,
        "max_samples": max_samples,
        "accuracy": accuracy,
        "elapsed_seconds": round(total_elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark and Evaluation for Task 1 Gated S3-TPT")
    parser.add_argument("--domain", choices=["clean", "adv", "r", "all"], default="clean",
                        help="Domain to evaluate: clean (Caltech101), adv (Caltech101 PGD), r (ImageNet-R), or all")
    parser.add_argument("--method", choices=["gated", "baseline", "both", "gated_k2"], default="gated",
                        help="Method to run: gated (Task 1 best solution), baseline (original SS-TPT), or both (head-to-head)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of test samples to evaluate (default: 2000 for full benchmark)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to run on (default: 0)")
    parser.add_argument("--weight", type=float, default=None,
                        help="Override spatial weight (default: 0.10 for gated, 0.0 for baseline)")
    parser.add_argument("--k", type=int, default=None,
                        help="Override number of weak augmentations tiny_k (default: 1)")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to store output logs and summary tables")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    domains_to_run = ["clean", "adv", "r"] if args.domain == "all" else [args.domain]
    methods_to_run = ["baseline", "gated"] if args.method == "both" else [args.method]

    results = []
    for d in domains_to_run:
        for m in methods_to_run:
            res = run_single_eval(d, m, args.samples, args.gpu, args.weight, args.k, out_dir)
            results.append(res)

    # Save results JSON
    timestamp = int(time.time())
    json_path = out_dir / f"eval_results_{args.domain}_{args.method}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n\n========================================================================================")
    print("                                FINAL BENCHMARK SUMMARY                                 ")
    print("========================================================================================")
    print(f"{'Domain':<30} | {'Method':<36} | {'Top-1 Acc':<10} | {'Time (s)':<10}")
    print("-" * 90)

    domain_groups = {}
    for r in results:
        d = r["domain"]
        if d not in domain_groups:
            domain_groups[d] = {}
        domain_groups[d][r["method_key"]] = r
        print(f"{r['domain']:<30} | {r['method']:<36} | {r['accuracy']:>6.2f}%    | {r['elapsed_seconds']:>8.1f}s")

    # If head-to-head comparison was run, calculate delta
    if "baseline" in methods_to_run and "gated" in methods_to_run:
        print("\n-------------------------------- HEAD-TO-HEAD COMPARISON -------------------------------")
        print(f"{'Domain':<30} | {'Baseline':<12} | {'Gated S3-TPT':<14} | {'Improvement (Delta)':<20}")
        print("-" * 90)
        for d, m_map in domain_groups.items():
            if "baseline" in m_map and "gated" in m_map:
                b_acc = m_map["baseline"]["accuracy"]
                g_acc = m_map["gated"]["accuracy"]
                delta = g_acc - b_acc
                delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
                print(f"{d:<30} | {b_acc:>6.2f}%     | {g_acc:>6.2f}%       | {delta_str:>12}")

    print("========================================================================================")
    print(f"Results saved to: {json_path}\n")


if __name__ == "__main__":
    main()

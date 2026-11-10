import subprocess
import os
import re
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

DATA_PATH = "/home/abhishek/workspace/SS-TPT/data"
PYTHON_BIN = "/home/abhishek/workspace/conda/sstpt/bin/python"

# Default to 2000 samples (matching comparison_2000.log) or take CLI arg
max_samples = 2000
force_rerun = False
gpu_id = 1

for arg in sys.argv[1:]:
    if arg.isdigit():
        max_samples = int(arg)
    elif arg == "--force":
        force_rerun = True
    elif arg.startswith("--gpu="):
        gpu_id = int(arg.split("=")[1])

experiments = [
    {
        "name": "Clean (Caltech101)",
        "domain": "Clean",
        "dataset": "Caltech101",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
        "max_samples": max_samples,
        "gpu": gpu_id
    },
    {
        "name": "Adversarial PGD (Caltech101)",
        "domain": "Adversarial",
        "dataset": "Caltech101",
        "eps": 1.0,
        "steps": 7,
        "alpha": 0.25,
        "max_samples": max_samples,
        "gpu": gpu_id
    },
    {
        "name": "Distribution-Shifted (ImageNet-R)",
        "domain": "Distribution-Shift",
        "dataset": "R",
        "eps": 0.0,
        "steps": 0,
        "alpha": 0.0,
        "max_samples": max_samples,
        "gpu": gpu_id
    }
]

methods = [
    {
        "method_name": "Original SS-TPT (2-Score Baseline)",
        "spat_weight": 0.0,
        "spat_method": "attn",
        "selection_mode": "entropy",
        "tiny_k": 1
    },
    {
        "method_name": "Proposed S3-TPT (Score-Guided Spatial)",
        "spat_weight": 0.20,
        "spat_method": "gradcam",
        "selection_mode": "score",
        "tiny_k": 1
    },
    {
        "method_name": "Improved S3-TPT (Attention-Guided Spatial)",
        "spat_weight": 0.20,
        "spat_method": "attn",
        "selection_mode": "entropy",
        "tiny_k": 1
    },
    {
        "method_name": "Gated S3-TPT (Attention, K=1, w=0.10)",
        "spat_weight": 0.10,
        "spat_method": "attn",
        "selection_mode": "entropy",
        "tiny_k": 1
    },
    {
        "method_name": "Gated S3-TPT (Attention, K=2, w=0.10)",
        "spat_weight": 0.10,
        "spat_method": "attn",
        "selection_mode": "entropy",
        "tiny_k": 2
    },
    {
        "method_name": "Proposed Task-1 S3-TPT (Aligned Pearson Spatial + Pareto)",
        "spat_weight": 0.15,
        "spat_method": "attn",
        "selection_mode": "pareto",
        "tiny_k": 1
    }
]

output_base_dir = "output_results/comparison_v2"
results = []

# Load previously verified 2000-sample results if available
cached_2000 = {}
for c_file in ["final_comparison_5way.json", "final_comparison_3way.json"]:
    cache_json = os.path.join(output_base_dir, c_file)
    if os.path.exists(cache_json):
        try:
            with open(cache_json, "r") as f:
                prev_results = json.load(f)
                for item in prev_results:
                    key = (item["domain"], item["method"])
                    cached_2000[key] = (item["accuracy"], item["eval_time_sec"])
        except Exception as e:
            print(f"Warning reading {c_file}:", e)

def get_out_dir(exp, m):
    safe_arch = "RN50"
    out_name = f"eps_{exp['eps']}_alpha_{exp['alpha']}_step_{exp['steps']}_batchsize_16"
    if m["spat_weight"] > 0.0:
        out_name += f"_spat_{m['spat_weight']}_{m['spat_method']}"
        if m.get("selection_mode", "entropy") != "entropy":
            out_name += f"_{m['selection_mode']}"
        if m.get("tiny_k", 1) > 1:
            out_name += f"_k{m['tiny_k']}"
    return os.path.join(output_base_dir, safe_arch, exp["dataset"], out_name)

def check_cached_result(exp, m):
    if exp["max_samples"] == 2000:
        key = (exp["domain"], m["method_name"])
        if key in cached_2000:
            return cached_2000[key]
            
    out_dir = get_out_dir(exp, m)
    log_path = os.path.join(out_dir, "log.txt")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        content = f.read()
    if f"max_samples:{exp['max_samples']}" not in content:
        return None
    acc_match = re.search(r"\*\s+SSTPTAcc@1\s+([\d\.]+)", content)
    if not acc_match:
        acc_match = re.search(r"=> Acc\. on testset \[.*?\]: SSTPT (?:Clean|Adv) Acc @1 ([\d\.]+)", content)
    time_match = re.search(r"Evaluation time:\s+([\d\.]+)\s+seconds", content)
    if acc_match and time_match:
        return float(acc_match.group(1)), float(time_match.group(1))
    return None

def save_current_results():
    os.makedirs(output_base_dir, exist_ok=True)
    with open(os.path.join(output_base_dir, "final_comparison_5way.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_base_dir, "final_comparison_crossview.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    for exp in experiments:
        for m in methods:
            cached = check_cached_result(exp, m) if not force_rerun else None
            
            if cached:
                acc, eval_time = cached
                print(f"\n[Cached 2000-sample Log] {exp['name']} with {m['method_name']}: Top-1 Acc = {acc:.2f}%, Time = {eval_time:.2f}s")
                results.append({
                    "domain": exp["domain"],
                    "dataset": exp["name"],
                    "method": m["method_name"],
                    "spat_weight": m["spat_weight"],
                    "selection_mode": m["selection_mode"],
                    "spat_method": m["spat_method"],
                    "tiny_k": m.get("tiny_k", 1),
                    "accuracy": acc,
                    "eval_time_sec": eval_time
                })
                save_current_results()
                continue

            print(f"\n=======================================================")
            print(f"Running {exp['name']} with {m['method_name']} (spat_weight={m['spat_weight']}, method={m['spat_method']}, K={m.get('tiny_k', 1)}, select={m['selection_mode']})...")
            print(f"=======================================================")
            
            cmd = [
                PYTHON_BIN, "-u", "sstpt.py", DATA_PATH,
                "-a", "RN50",
                "-b", "16",
                "--gpu", str(exp["gpu"]),
                "--ctx_init", "a_photo_of_a",
                "-p", "25",
                "--eps", str(exp["eps"]),
                "--steps", str(exp["steps"]),
                "--alpha", str(exp["alpha"]),
                "--test_sets", exp["dataset"],
                "--output_dir", output_base_dir,
                "--max_samples", str(exp["max_samples"]),
                "--spat_weight", str(m["spat_weight"]),
                "--spat_method", m["spat_method"],
                "--selection_mode", m["selection_mode"],
                "--tiny_k", str(m.get("tiny_k", 1))
            ]
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            full_output = []
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                full_output.append(line)
            proc.wait()
            output = "".join(full_output)
            
            match = re.search(r"\*\s+SSTPTAcc@1\s+([\d\.]+)", output)
            if match:
                acc = float(match.group(1))
            else:
                match_iter = re.findall(r"sstpt_(?:acc|adv)1=([\d\.]+)", output)
                acc = float(match_iter[-1]) if match_iter else 0.0
                
            time_match = re.search(r"Evaluation time:\s+([\d\.]+)\s+seconds", output)
            eval_time = float(time_match.group(1)) if time_match else 0.0
            
            results.append({
                "domain": exp["domain"],
                "dataset": exp["name"],
                "method": m["method_name"],
                "spat_weight": m["spat_weight"],
                "selection_mode": m["selection_mode"],
                "spat_method": m["spat_method"],
                "tiny_k": m.get("tiny_k", 1),
                "accuracy": acc,
                "eval_time_sec": eval_time
            })
            save_current_results()

    print("\n\n================ FINAL COMPLETE COMPARISON SUMMARY ================")
    print(f"{'Domain':<18} | {'Method':<42} | {'Top-1 Acc':<10} | {'vs Base':<9} | {'Time (s)':<10}")
    print("-" * 102)
    baselines = {}
    for r in results:
        if "Baseline" in r['method']:
            baselines[r['domain']] = r['accuracy']
            
    for r in results:
        base_acc = baselines.get(r['domain'], None)
        if base_acc is not None and "Baseline" not in r['method']:
            diff = r['accuracy'] - base_acc
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}%"
        else:
            diff_str = "  baseline"
        print(f"{r['domain']:<18} | {r['method']:<42} | {r['accuracy']:>6.2f}%    | {diff_str:>10} | {r['eval_time_sec']:>7.2f}s")

    save_current_results()

import json
import os

files = {
    "Clean": "task1_exp_clean.json",
    "Adversarial": "task1_exp_adv.json",
    "Distribution-Shift": "task1_exp_r.json"
}

data = {}
for domain, fpath in files.items():
    if os.path.exists(fpath):
        with open(fpath) as f:
            data[domain] = json.load(f)

if not data:
    print("No exploration results found yet.")
    exit()

domains = list(data.keys())
cfgs = list(data[domains[0]].keys())

print("=" * 105)
print("TASK 1 COMPREHENSIVE EXPLORATION RESULTS (Accuracy & vs Baseline)")
print("=" * 105)
header = f"{chr(39)}Configuration{chr(39):<27}"
header = f"{header:<28}"
for d in domains:
    header += f" | {d:<20}"
header += " | Combined Delta"
print(header)
print("-" * 105)

for c in cfgs:
    row = f"{c:<28}"
    net_diff = 0.0
    valid_count = 0
    for d in domains:
        if c in data[d]:
            acc = data[d][c]["acc"]
            diff = data[d][c]["diff"]
            net_diff += diff
            valid_count += 1
            sign = "+" if diff >= 0 else ""
            if c == "baseline":
                row += f" | {acc:.2f}% (base)        "
            else:
                row += f" | {acc:.2f}% ({sign}{diff:.2f}%)     "
        else:
            row += f" | {chr(39)}N/A{chr(39):<20}"
    sign_net = "+" if net_diff >= 0 else ""
    row += f" | {sign_net}{net_diff:.2f}%"
    print(row)
print("=" * 105)

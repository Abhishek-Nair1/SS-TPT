import json
import os

files = {
    "Clean": "val_results_clean.json",
    "Adversarial": "val_results_adv.json",
    "Distribution-Shift": "val_results_r.json"
}

data = {}
for domain, fpath in files.items():
    if os.path.exists(fpath):
        with open(fpath) as f:
            data[domain] = json.load(f)

if not data:
    print("No results found yet.")
    exit()

domains = list(data.keys())
cfgs = list(data[domains[0]].keys())

print("=" * 90)
print("VALIDATION SWEEP RESULTS (Accuracy & vs Baseline)")
print("=" * 90)
header = f"{'Configuration':<25}"
for d in domains:
    header += f" | {d:<18}"
print(header)
print("-" * 90)

for c in cfgs:
    row = f"{c:<25}"
    for d in domains:
        if c in data[d]:
            acc = data[d][c]["acc"]
            diff = data[d][c]["diff"]
            sign = "+" if diff >= 0 else ""
            if c == "baseline":
                row += f" | {acc:.2f}% (base)    "
            else:
                row += f" | {acc:.2f}% ({sign}{diff:.2f}%)  "
        else:
            row += f" | {'N/A':<18}"
    print(row)
print("=" * 90)

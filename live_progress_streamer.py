#!/usr/bin/env python3
"""Streams clean, real-time benchmark progress into master_benchmark_pragyan_2000.log."""

import os
import time
import re
from datetime import datetime

LOG_FILE = "/home/abhishek/workspace/Final_Project_Folder/master_benchmark_pragyan_2000.log"
BASE_DIR = "/home/abhishek/workspace/Final_Project_Folder/output_results"

JOBS = [
    ("GPU 0", "Combined Clean", "combined/clean"),
    ("GPU 0", "Combined Shift", "combined/shift"),
    ("GPU 0", "Adaptive Alpha Shift", "adaptive_alpha/shift"),
    ("GPU 1", "Combined Adversarial", "combined/adversarial"),
    ("GPU 2", "Adaptive Alpha Adversarial", "adaptive_alpha/adversarial"),
]

def get_job_info(subpath):
    d = os.path.join(BASE_DIR, subpath)
    trace_f = os.path.join(d, "alpha_trace.jsonl")
    out_f = os.path.join(d, "output.txt")
    if not os.path.exists(d):
        return "Pending", 0, 0.0

    cnt = 0
    if os.path.exists(trace_f):
        try:
            with open(trace_f) as f:
                cnt = sum(1 for _ in f)
        except Exception:
            pass

    acc = 0.0
    if os.path.exists(out_f):
        try:
            with open(out_f) as f:
                lines = f.readlines()
            for l in reversed(lines):
                m = re.search(r"acc1=([\d\.]+)%", l)
                if m:
                    acc = float(m.group(1))
                    break
                m2 = re.search(r"Top-1 Accuracy:\s+([\d\.]+)%", l)
                if m2:
                    acc = float(m2.group(1))
                    break
        except Exception:
            pass

    if cnt >= 2000:
        return "Done", cnt, acc
    elif cnt > 0:
        return "Running", cnt, acc
    else:
        return "Pending", cnt, acc

def main():
    last_status = {}
    while True:
        all_done = True
        updates = []
        for gpu, name, subpath in JOBS:
            status, cnt, acc = get_job_info(subpath)
            if status != "Done":
                all_done = False
            prev = last_status.get(name)
            if status == "Done" and prev != "Done":
                updates.append(f"[{gpu}] <<< FINISHED: {name} (Top-1 Acc: {acc:.2f}%)")
            elif status == "Running":
                pct = (cnt / 2000.0) * 100
                updates.append(f"[{gpu} | {name}] {cnt:>4}/2000 ({pct:>5.1f}%) | acc1={acc:.2f}%")
            last_status[name] = status

        if updates:
            t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a") as f:
                for u in updates:
                    f.write(f"[{t_str}] {u}\n")
                f.flush()

        if all_done:
            t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, "a") as f:
                f.write(f"\n[{t_str}] ================================================================================\n")
                f.write(f"[{t_str}] ALL BENCHMARKS COMPLETED!\n")
                f.write(f"[{t_str}] ================================================================================\n")
                f.flush()
            break

        time.sleep(15)

if __name__ == "__main__":
    main()


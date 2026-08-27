#!/usr/bin/env python3
"""One-off: rerun t22 (1 input) and append to results/standard_results.csv."""
import csv, re, subprocess, time

csvf = "suite-standard/bench_t22_windowed_sine.csv"
args = ["..\\aria10.exe", "--csv", csvf, "--input-cols", "1",
        "--max-epochs", "100", "--seed", "1", "--save-graph", "none",
        "--eval-csv", csvf]
t0 = time.time()
with open("standard_t22_windowed_sine_run.txt", "w",
          encoding="utf-8", errors="replace") as lf:
    subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
text = open("standard_t22_windowed_sine_run.txt", encoding="utf-8",
            errors="replace").read()
m = re.findall(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)", text)
com = re.findall(r"COMMIT rank=\d+ type=(\w+)", text)
print(f"t22: R2s={m} commits={com} ({time.time()-t0:.0f}s)", flush=True)
if m:
    rows = list(csv.reader(open("results/standard_results.csv")))
    for i, r in enumerate(rows):
        if r[0] == "t22_windowed_sine":
            worst = min(float(v) for _, v in m)
            r[1] = f"{worst:.6f}"
            r[2] = " ".join(f"o{k}={v}" for k, v in m)
            r[3] = ",".join(com) or "none"
            r[4] = round(time.time() - t0, 1)
    with open("results/standard_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("results/standard_results.csv updated", flush=True)

#!/usr/bin/env python3
"""Determinism probe: run hetero3 N times on one exe, compare per-output R2.
Usage: python det_probe.py <exe> <tag> [runs]"""
import os, re, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")

wd = os.path.join("detprobe", TAG)
os.makedirs(wd, exist_ok=True)
csv_path = os.path.abspath("suite-limits/bench_hetero3.csv")
for i in range(RUNS):
    args = [EXE, "--csv", csv_path, "--input-cols", "1", "--output-cols", "3",
            "--max-epochs", "100", "--seed", "1", "--save-graph", "none",
            "--eval-csv", csv_path]
    log = os.path.join(wd, f"hetero3_run{i}.txt")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd,
                       timeout=3600)
    text = open(log, encoding="utf-8", errors="replace").read()
    r2s = {int(a) + 1: float(b) for a, b in R2A_RE.findall(text)}
    per = " ".join(f"o{k}={v:.4f}" for k, v in sorted(r2s.items())) or "no-eval"
    print(f"run{i}: {per} ({time.time()-t0:.0f}s)", flush=True)

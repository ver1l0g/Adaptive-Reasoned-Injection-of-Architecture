#!/usr/bin/env python3
"""M1.3 fatigue A/B: <exe> <tag> — d7, hetero3, stripes20, t23, t31."""
import os, re, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
TASKS = [
    ("d7_multiout", 1, 3, "suite-standard/bench_d7_multiout.csv", []),
    ("hetero3", 1, 3, "suite-limits/bench_hetero3.csv", []),
    ("stripes20", 1, 1, "suite-limits/bench_stripes20.csv", []),
    ("t23_quadratic", 2, 1, "suite-standard/bench_t23_quadratic.csv", []),
    ("t31_threeway", 3, 1, "suite-standard/bench_t31_three_way_product.csv", []),
]
R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")
FATIGUE_RE = re.compile(r"M1\.3 family fatigue: suppressing (\w+)")

wd = os.path.join("bisect", TAG)
os.makedirs(wd, exist_ok=True)
for name, nin, nout, csvf, extra in TASKS:
    src = os.path.abspath(csvf)
    args = [EXE, "--csv", src, "--input-cols", str(nin), "--output-cols", str(nout),
            "--max-epochs", "100", "--seed", "1", "--save-graph", "none",
            "--eval-csv", src] + extra
    log = os.path.join(wd, f"{name}.txt")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd, timeout=1800)
    text = open(log, encoding="utf-8", errors="replace").read()
    r2s = {int(a) + 1: float(b) for a, b in R2A_RE.findall(text)}
    fat = ",".join(FATIGUE_RE.findall(text)) or "none"
    per = " ".join(f"o{k}={v:.4f}" for k, v in sorted(r2s.items())) or "no-eval"
    print(f"{name:<15} {per}  fatigue=[{fat}]  ({time.time()-t0:.0f}s)", flush=True)

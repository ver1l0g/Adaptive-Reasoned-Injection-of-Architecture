#!/usr/bin/env python3
"""Zero-plateau A/B: t22 (target) + t21/d1 (plateau tasks) + d9 (must not fire)."""
import os, re, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
TASKS = [
    ("t22_windowed_sine", 1, "suite-standard/bench_t22_windowed_sine.csv"),
    ("t21_three_region", 1, "suite-standard/bench_t21_three_region.csv"),
    ("d1_step", 1, "suite-standard/bench_d1_step.csv"),
    ("d9_noise", 2, "suite-standard/bench_d9_noise.csv"),
]
R2_RE = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")
ZP_RE = re.compile(r"Zero-plateau boundary: thr=([-\d.]+)")

wd = os.path.join("bisect", TAG)
os.makedirs(wd, exist_ok=True)
for name, nin, csvf in TASKS:
    src = os.path.abspath(csvf)
    args = [EXE, "--csv", src, "--input-cols", str(nin),
            "--max-epochs", "100", "--seed", "1", "--save-graph", "none",
            "--eval-csv", src]
    log = os.path.join(wd, f"{name}.txt")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd,
                       timeout=1800)
    text = open(log, encoding="utf-8", errors="replace").read()
    m = R2_RE.search(text)
    r2 = float(m.group(1)) if m else None
    zps = ["%.3f" % float(z) for z in ZP_RE.findall(text)]
    tag = "PASS" if r2 and r2 > 0.99 else "FAIL"
    print(f"{name:<20} R2={r2 if r2 is None else round(r2,4)} [{tag}] "
          f"plateau_thrs={zps or 'none'} ({time.time()-t0:.0f}s)", flush=True)

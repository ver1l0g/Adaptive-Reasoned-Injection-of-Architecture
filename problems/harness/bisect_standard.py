#!/usr/bin/env python3
"""Bisect the standard-suite regression (d2 d3 d7 t21 t22) across
binaries and library states. Runs each battery in an isolated workdir
(fresh library) unless 'copylib' is passed (replicates the polluted
shared library + failure library from problems/).

Usage: python bisect_standard.py <exe> <tag> [copylib]
Writes bisect_<tag>/results.csv + per-task logs; prints progress.
"""
import csv, os, re, shutil, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1])
TAG = sys.argv[2]
COPY_LIB = len(sys.argv) > 3 and sys.argv[3] == "copylib"

TASKS = [
    ("d2_mod3", 1, 1, []),
    ("d3_two_sines", 1, 1, []),
    ("d7_multiout", 1, 3, []),
    ("t21_three_region", 1, 1, []),
    ("t22_windowed_sine", 1, 1, []),
]

wd = os.path.join("bisect", TAG)
os.makedirs(wd, exist_ok=True)
if COPY_LIB:
    for fn in ("subgraph_library.txt", "failure_library.txt"):
        if os.path.exists(fn):
            shutil.copy2(fn, os.path.join(wd, fn))
            print(f"copied {fn} ({os.path.getsize(fn)}B)", flush=True)

R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")

with open(os.path.join(wd, "results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "min_r2", "per_out", "wall_s"])
    f.flush()
    for i, (name, nin, nout, extra) in enumerate(TASKS):
        src = os.path.abspath(os.path.join("suite-standard", f"bench_{name}.csv"))
        args = [EXE, "--csv", src, "--input-cols", str(nin)]
        if nout > 1:
            args += ["--output-cols", str(nout)]
        args += ["--max-epochs", "100", "--seed", "1", "--save-graph",
                 "none", "--eval-csv", src] + extra
        log = os.path.join(wd, f"{name}.txt")
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as lf:
                subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT,
                               cwd=wd, timeout=1800)
        except subprocess.TimeoutExpired:
            pass
        text = open(log, encoding="utf-8", errors="replace").read()
        r2s = {int(a) + 1: float(b) for a, b in R2A_RE.findall(text)}
        wall = round(time.time() - t0, 1)
        if r2s:
            worst = min(r2s.values())
            per = " ".join(f"o{k}={v:.4f}" for k, v in sorted(r2s.items()))
            tag = "PASS" if worst > 0.99 else "FAIL"
        else:
            worst, per, tag = None, "no-eval", "FAIL"
        w.writerow([name, "" if worst is None else f"{worst:.6f}", per, wall])
        f.flush()
        r2t = f"R2={worst:.4f}" if worst is not None else "FAIL"
        print(f"[{i+1}/5] {name:<18} {r2t} [{tag}] ({wall}s)", flush=True)
print(f"bisect {TAG} done", flush=True)

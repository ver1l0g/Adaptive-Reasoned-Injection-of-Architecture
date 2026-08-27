#!/usr/bin/env python3
"""Epoch-sensitivity probe: do d2/d3/t22 pass with more budget on aria10?"""
import re, subprocess, sys, time

TASKS = [("d2_mod3", 1, 1), ("d3_two_sines", 1, 1), ("t22_windowed_sine", 1, 1)]
EPOCHS = sys.argv[1] if len(sys.argv) > 1 else "400"
R2_RE = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")

for name, nin, nout in TASKS:
    csvf = f"suite-standard/bench_{name}.csv"
    args = ["..\\aria10.exe", "--csv", csvf, "--input-cols", str(nin),
            "--max-epochs", EPOCHS, "--seed", "1", "--save-graph", "none",
            "--eval-csv", csvf]
    log = f"probe_{name}_e{EPOCHS}.txt"
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    text = open(log, encoding="utf-8", errors="replace").read()
    m = R2_RE.search(text)
    r2 = float(m.group(1)) if m else None
    tag = f"R2={r2:.4f}" if r2 is not None else "FAIL"
    verdict = "PASS" if r2 and r2 > 0.99 else "FAIL"
    print(f"{name:<18} {tag} [{verdict}] ({time.time()-t0:.0f}s)", flush=True)

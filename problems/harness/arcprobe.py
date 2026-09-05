#!/usr/bin/env python3
"""Arc-price test bed: hetero3 (sin arc = investment), stripes20 (TANH spray
= waste), d2 (harmonic ladder = investment). Prints each commit's
fingerprint_move to verify the signal separates investment from spray."""
import os, re, subprocess, sys, time

EXE = os.path.abspath(r"..\aria27.exe")
TASKS = [
    ("hetero3", 1, 3, r"suite-limits\bench_hetero3.csv"),
    ("stripes20", 1, 1, r"suite-limits\bench_stripes20.csv"),
    ("d2_mod3", 1, 1, r"suite-standard\bench_d2_mod3.csv"),
]
ARC_RE = re.compile(r"\[ARC-PRICE\] family=(\S+) fingerprint_move=([\d.]+) val_delta=([-\d.e+]+)")
R2_RE = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")

for name, nin, nout, csvf in TASKS:
    wd = os.path.join("arcprobe", name)
    os.makedirs(wd, exist_ok=True)
    src = os.path.abspath(csvf)
    args = [EXE, "--csv", src, "--input-cols", str(nin)]
    if nout > 1:
        args += ["--output-cols", str(nout)]
    args += ["--max-epochs", "150", "--seed", "1", "--save-graph", "none",
             "--eval-csv", src]
    log = os.path.join(wd, "log.txt")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd,
                       timeout=2400)
    text = open(log, encoding="utf-8", errors="replace").read()
    print(f"=== {name} ({(time.time()-t0)/60:.0f}min) ===")
    for m in ARC_RE.finditer(text):
        print(f"  {m.group(1):<24} move={m.group(2):<10} val_delta={m.group(3)}")
    r2 = R2_RE.search(text)
    print(f"  final R2={r2.group(1) if r2 else 'n/a'}")

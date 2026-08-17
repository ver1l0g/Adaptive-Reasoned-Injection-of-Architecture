#!/usr/bin/env python3
"""Temporal memory harness: lorenz3 (control) -> lorenz -> narma10_lag -> narma10.
Run logs tee'd to temporal_<task>_run.txt (crash-safe parsing)."""
import re, subprocess, sys, time
from collections import Counter

EXE = sys.argv[1] if len(sys.argv) > 1 else "../gpnn7.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

TASKS = [
    ("lorenz3",     ["--csv","bench_lorenz3.csv","--input-cols","3","--output-cols","3"]),
    ("lorenz",      ["--csv","bench_lorenz.csv","--input-cols","1"]),
    ("narma10_lag", ["--csv","bench_narma10_lag.csv","--input-cols","10"]),
    ("narma10",     ["--csv","bench_narma10.csv","--input-cols","1"]),
]

VAL_RE  = re.compile(r"Restored best-val graph snapshot \(val=([0-9.eE+-]+)")
LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")
COM_RE  = re.compile(r"COMMIT rank=\d+ type=(\w+)")
BESTVAL_RE = re.compile(r"New best val loss: ([0-9.eE+-]+)")
EPOCH_RE   = re.compile(r"sgd\s+ loss=([0-9.eE+-]+)")

def parse_run(logpath):
    try:
        text = open(logpath, encoding="utf-8", errors="replace").read()
    except OSError:
        return None, None, "no-log"
    commits = ",".join(f"{k}x{v}" for k, v in Counter(COM_RE.findall(text)).items()) or "none"
    vm = VAL_RE.search(text)
    val = float(vm.group(1)) if vm else None
    if val is None:
        bm = BESTVAL_RE.findall(text)
        val = float(bm[-1]) if bm else None
    lm = LOSS_RE.search(text)
    loss = float(lm.group(1)) if lm else None
    if loss is None:
        em = EPOCH_RE.findall(text)
        loss = float(em[-1]) if em else None
    return val, loss, commits

def main():
    for name, args in TASKS:
        t0 = time.time()
        logpath = f"temporal_{name}_run.txt"
        full = [EXE] + args + ["--no-shuffle", "--max-epochs", str(EPOCHS),
                               "--seed", "1", "--save-graph", "none"]
        try:
            with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                subprocess.run(full, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=3600)
        except subprocess.TimeoutExpired:
            pass
        val, loss, commits = parse_run(logpath)
        print(f"{name:<12} val={val if val is not None else 'n/a':<12} "
              f"train={loss if loss is not None else 'n/a':<12} commits={commits} "
              f"({round(time.time()-t0,0):.0f}s)", flush=True)

if __name__ == "__main__":
    main()

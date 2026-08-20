#!/usr/bin/env python3
"""Limit-probe harness: 7 probes, crash-safe tee'd logs, R2-based verdicts."""
import re, subprocess, sys, time

EXE = sys.argv[1] if len(sys.argv) > 1 else "../aria5.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

TASKS = [
    ("highdim15", ["--csv","bench_highdim15.csv","--input-cols","15"], 0.99),
    ("highdim20", ["--csv","bench_highdim20.csv","--input-cols","20"], 0.99),
    ("narma30",   ["--csv","bench_narma30.csv","--input-cols","1","--no-shuffle"], 0.99),
    ("hetero3",   ["--csv","bench_hetero3.csv","--input-cols","1","--output-cols","3"], 0.99),
    ("stripes20", ["--csv","bench_stripes20.csv","--input-cols","1"], 0.95),
    ("count8",    ["--csv","bench_count8.csv","--input-cols","8"], 0.99),
    ("firstpos8", ["--csv","bench_firstpos8.csv","--input-cols","8"], 0.99),
]

R2_RE  = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")
R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")
LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")
COM_RE = re.compile(r"COMMIT rank=\d+ type=(\w+)")

def main():
    for name, args, thresh in TASKS:
        t0 = time.time()
        logpath = f"limits_{name}_run.txt"
        full = [EXE] + args + ["--max-epochs", str(EPOCHS), "--seed", "1",
                               "--save-graph", "none"]
        # eval on train file (probes are about capability, not generalization)
        csv_arg = next(a for a in args if a.endswith(".csv"))
        full += ["--eval-csv", csv_arg]
        try:
            with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                subprocess.run(full, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=1800)
        except subprocess.TimeoutExpired:
            pass
        try:
            text = open(logpath, encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        from collections import Counter
        commits = ",".join(f"{k}x{v}" for k, v in Counter(COM_RE.findall(text)).items()) or "none"
        r2s = {int(m.group(1)) + 1: float(m.group(2)) for m in R2A_RE.finditer(text)}
        verdict = []
        for out_i, r2 in sorted(r2s.items()):
            ok = "PASS" if r2 > thresh else "FAIL"
            verdict.append(f"out{out_i}={r2:.4f}[{ok}]")
        vstr = " ".join(verdict) if verdict else "no-eval"
        print(f"{name:<11} {vstr}  commits={commits}  ({(time.time()-t0)/60:.0f}min)",
              flush=True)

if __name__ == "__main__":
    main()

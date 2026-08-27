#!/usr/bin/env python3
"""Standard-suite harness (17 tasks): crash-safe tee'd logs, per-output R2.

Mirrors run_korns.py / run_limits.py conventions. Eval on the train file
(capability probe, identical to run_limits). d9 is the noise-floor task:
its R2 is reported but flagged 'NF' instead of PASS/FAIL.
"""
import csv, re, subprocess, sys, time
from collections import Counter

EXE = sys.argv[1] if len(sys.argv) > 1 else "../aria10.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# (name, input-cols, output-cols, extra args, noise-floor task?)
TASKS = [
    ("d1_step",      1, 1, [],                    False),
    ("d2_mod3",      1, 1, [],                    False),
    ("d3_two_sines", 1, 1, [],                    False),
    ("d4_irrelevant12", 12, 1, [],                False),
    ("d5_extrap_cliff", 1, 1, [],                 False),
    ("d6_seq_parity", 1, 1, ["--no-shuffle"],     False),
    ("d7_multiout",  1, 3, [],                    False),
    ("d8_compose_absprod", 2, 1, [],              False),
    ("d9_noise",     2, 1, [],                    True),
    ("t21_three_region", 1, 1, [],                False),
    ("t22_windowed_sine", 2, 1, [],               False),
    ("t23_quadratic", 2, 1, [],                   False),
    ("t24_quadrant_xor", 2, 1, [],                False),
    ("t31_three_way_product", 3, 1, [],           False),
    ("t32_sine_of_product", 2, 1, [],             False),
    ("t33_absolute_value", 1, 1, [],              False),
    ("t34_xor3_parity", 3, 1, [],                 False),
]

R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")
COM_RE = re.compile(r"COMMIT rank=\d+ type=(\w+)")

def main():
    results = []
    with open("results/standard_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "min_r2", "r2_per_out", "commits", "wall_s"])
        for i, (name, nin, nout, extra, nf) in enumerate(TASKS):
            t0 = time.time()
            csvf = f"suite-standard/bench_{name}.csv"
            args = [EXE, "--csv", csvf, "--input-cols", str(nin)]
            if nout > 1:
                args += ["--output-cols", str(nout)]
            args += ["--max-epochs", str(EPOCHS), "--seed", "1",
                     "--save-graph", "none", "--eval-csv", csvf] + extra
            logpath = f"standard_{name}_run.txt"
            try:
                with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                    subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=1800)
            except subprocess.TimeoutExpired:
                pass
            try:
                text = open(logpath, encoding="utf-8", errors="replace").read()
            except OSError:
                text = ""
            commits = ",".join(f"{k}x{v}" for k, v in
                               Counter(COM_RE.findall(text)).items()) or "none"
            r2s = {int(m.group(1)) + 1: float(m.group(2))
                   for m in R2A_RE.finditer(text)}
            if r2s:
                worst = min(r2s.values())
                per_out = " ".join(f"o{k}={v:.4f}" for k, v in sorted(r2s.items()))
                tag = "NF" if nf else ("PASS" if worst > 0.99 else "FAIL")
            else:
                worst, per_out, tag = None, "no-eval", "FAIL"
            wall = round(time.time() - t0, 1)
            results.append(worst)
            w.writerow([name, "" if worst is None else f"{worst:.6f}",
                        per_out, commits, wall])
            f.flush()
            r2txt = f"R2={worst:.4f}" if worst is not None else "FAIL"
            print(f"[{i+1}/17] {name:<24} {r2txt} [{tag}]  ({wall}s)", flush=True)

    ok = [r for r in results if r is not None]
    solved = sum(1 for r in ok if r > 0.99)
    print(f"\n=== Standard: solved (min-out R2>0.99, d9 excluded as NF): "
          f"{solved}/16 + d9(NF) ===")

if __name__ == "__main__":
    main()

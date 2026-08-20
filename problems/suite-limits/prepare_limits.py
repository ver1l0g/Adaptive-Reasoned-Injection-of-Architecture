#!/usr/bin/env python3
"""Limit-probe battery: five synthetic probes targeting unknown failure modes.

  highdim15/20 — sparse nonlinear fn of 15/20 vars (poly-fit O(F^2) scaling)
  narma30      — 30-step memory (where does k=1 BPTT compensation break?)
  hetero3      — one input, three outputs needing DIFFERENT structures
                 (linear + periodic + boolean) — multi-diagnosis divergence
  stripes20    — 1D 20-region piecewise (IFELSE commit-count scaling)
  count8       — window count (linear sanity check; should solve)
  firstpos8    — index of first 1 in window (indexing; predicted impossible
                 without attention-class machinery)
"""
import csv, math, os, random

def write(fname, header, rows):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  {fname}: {len(rows)} rows")

def highdim(F, seed=42, n=1200):
    rng = random.Random(seed)
    # Sparse: uses inputs 0,1,2,3,4 only; rest irrelevant
    def fn(xs):
        return (xs[0]*xs[1] + math.sin(2*xs[2]) + xs[3]**2
                - 0.5*xs[4]*math.sin(xs[0]) + 1.0)
    rows = []
    for _ in range(n):
        xs = [rng.uniform(-2, 2) for _ in range(F)]
        rows.append([f"{v:.6f}" for v in xs] + [f"{fn(xs):.6f}"])
    write(f"bench_highdim{F}.csv", [f"x{i}" for i in range(F)] + ["y"], rows)

def narma30(n=1500, warmup=300, seed=42):
    rng = random.Random(seed)
    M = 30
    u = [rng.uniform(0.0, 0.5) for _ in range(n + warmup + M + 1)]
    y = [0.0] * (n + warmup + M + 1)
    for t in range(M, n + warmup + M):
        s = sum(y[t-i] for i in range(M))       # y[t-1..t-29]
        y[t+1] = (0.2*y[t] + (0.04/M)*y[t]*s
                  + 1.5*u[t-(M-1)]*u[t] + 0.01)
    rows = []
    for t in range(warmup + M + 1, warmup + M + 1 + n):
        rows.append([f"{u[t]:.6f}", f"{y[t]:.6f}"])
    write("bench_narma30.csv", ["u", "y"], rows)

def hetero3(n=1200, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.uniform(-3, 3)
        y0 = 2.0*x + 1.0
        y1 = math.sin(3.0*x)
        y2 = 1.0 if x > 0 else 0.0
        rows.append([f"{x:.6f}", f"{y0:.6f}", f"{y1:.6f}", f"{y2:.6f}"])
    write("bench_hetero3.csv", ["x", "y_lin", "y_sin", "y_bool"], rows)

def stripes20(n=1200, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.uniform(-1, 1)
        y = float((math.floor(x * 10)) % 2)     # 20 stripes
        rows.append([f"{x:.6f}", f"{y:.6f}"])
    write("bench_stripes20.csv", ["x", "y"], rows)

def window_probes(n=1500, seed=42, W=8):
    rng = random.Random(seed)
    cnt_rows, pos_rows = [], []
    for _ in range(n):
        bits = [rng.randint(0, 1) for _ in range(W)]
        count = float(sum(bits)) / W            # normalized count (linear)
        first = float(next((i for i, b in enumerate(bits) if b), W)) / W
        hdr = [f"b{i}" for i in range(W)]
        cnt_rows.append([str(b) for b in bits] + [f"{count:.6f}"])
        pos_rows.append([str(b) for b in bits] + [f"{first:.6f}"])
    write("bench_count8.csv", hdr + ["y"], cnt_rows)
    write("bench_firstpos8.csv", hdr + ["y"], pos_rows)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Limit probes:")
    highdim(15)
    highdim(20)
    narma30()
    hetero3()
    stripes20()
    window_probes()
    print("done — narma30 with --no-shuffle")

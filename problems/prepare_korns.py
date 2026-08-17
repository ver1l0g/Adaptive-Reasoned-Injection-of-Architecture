#!/usr/bin/env python3
"""Korns benchmark (9 canonical GP problems) for GP-NN.

Standard Korns configuration: 5 inputs x0..x4, x0..x3 uniform in [-3,3],
x4 uniform in [0,100] (an irrelevant distractor variable). 500 train /
300 test rows, noise-free.

F1..F9 in increasing difficulty. The ^0.5 family (F2, F5-F9) requires
SQRT of possibly-negative values — canonical Korns uses |arg|^0.5.
"""
import csv, math, os, random

FNS = [
    ("F1", 5, lambda x: 2.3*x[0] - 1.7*x[1]**2),
    ("F2", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs(x[1]))),
    ("F3", 5, lambda x: 2.3*x[0] - 1.7*x[1]*math.sin(x[0])),
    ("F4", 5, lambda x: 2.3*x[0] - 1.7*x[1]*math.sin(x[0]*x[1])),
    ("F5", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs(x[1]*x[2]))),
    ("F6", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs(x[1]*math.sin(x[2])))),
    ("F7", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs(x[1]**3 * x[2]))),
    ("F8", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs(x[1]*math.sin(x[0])*x[2]**3*x[3]))),
    ("F9", 5, lambda x: 2.3*x[0] - 1.7*math.sqrt(abs((x[1]-x[0])**3 * (x[2]-x[0]) * math.sqrt(abs(x[3]-x[0]))))),
]

def main(seed=42):
    os.makedirs("korns", exist_ok=True)
    rng = random.Random(seed)
    for fid, nvars, fn in FNS:
        header = [f"x{i}" for i in range(nvars)] + ["y"]
        with open(f"korns/{fid}.csv", "w", newline="") as ftr, \
             open(f"korns/{fid}_test.csv", "w", newline="") as fte:
            wtr, wte = csv.writer(ftr), csv.writer(fte)
            wtr.writerow(header); wte.writerow(header)
            for i in range(800):
                xs = [rng.uniform(-3, 3) for _ in range(4)] + [rng.uniform(0, 100)]
                y = fn(xs)
                row = [f"{v:.6f}" for v in xs] + [f"{y:.6f}"]
                (wtr if i < 500 else wte).writerow(row)
    print(f"Wrote {len(FNS)} Korns problems -> korns/ (500 train + 300 test)")

if __name__ == "__main__":
    main()

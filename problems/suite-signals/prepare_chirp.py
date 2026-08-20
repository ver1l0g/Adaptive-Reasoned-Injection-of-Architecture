"""Generate chirp signal sin(x^2) for GP-NN.

The chirp's frequency increases with x — the function's character changes
across the domain. A single SIN_INJECTION gives a fixed frequency; fitting
a chirp requires either spatially-varying frequency or domain-split IFELSE
+ per-region SIN commits.

y = sin(x^2), x in [0, 4]  (frequency ranges from 0 to ~5 cycles)
"""
import math, random

random.seed(42)

with open("bench_chirp.csv", "w") as f:
    for _ in range(300):
        x = random.uniform(0, 4)
        y = math.sin(x * x)
        f.write(f"{x:.6f},{y:.6f}\n")

print("Wrote bench_chirp.csv: 300 samples, 1 input, 1 output")
print("Range: x in [0,4], y = sin(x^2) in [-1,1]")
print("Frequency at x=0: 0 cycles, at x=4: ~5 cycles (chirp)")

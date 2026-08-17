#!/usr/bin/env python3
"""Prepare hard benchmark datasets that stress-test structural search.

Each problem requires a different architectural capability:
  - Two Spirals: requires deep nonlinear boundaries (multiple hidden layers)
  - Concentric Circles: requires x1²+x2² feature combination
  - Checkerboard: requires x1*x2 interaction with thresholding
  - 3D Sine Product: y=sin(x1*x2*x3) — needs 3-way compound hypothesis
  - Mixed Polynomial: y=x1²+x2²-x1*x2*x3+sin(x4) — multiple feature types
  - Spiral Regression: continuous spiral — tests sin/cos decomposition
"""
import csv, math, os, random

def write_csv(filename, header, rows):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {filename}: {len(rows)} samples")

def spiral(N=300, seed=42):
    """Two interleaving spirals — classic hard classification."""
    rng = random.Random(seed)
    rows = []
    points_per_spiral = N // 2
    for cls in range(2):
        for i in range(points_per_spiral):
            t = 2.5 * i / points_per_spiral  # 0 to 2.5 turns
            r = 0.3 + t * 0.8
            theta = t * 2 * math.pi + cls * math.pi
            x1 = r * math.cos(theta) + rng.gauss(0, 0.05)
            x2 = r * math.sin(theta) + rng.gauss(0, 0.05)
            rows.append([f"{x1:.4f}", f"{x2:.4f}", f"{cls}"])
    random.Random(seed).shuffle(rows)
    write_csv('bench_spirals.csv', ['x1','x2','target'], rows)

def circles(N=300, seed=42):
    """Concentric circles — requires x1²+x2² radial feature."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        cls = rng.randint(0, 2)
        r = 0.5 + cls * 0.8 + rng.gauss(0, 0.1)
        theta = rng.uniform(0, 2*math.pi)
        x1 = r * math.cos(theta)
        x2 = r * math.sin(theta)
        rows.append([f"{x1:.4f}", f"{x2:.4f}", f"{cls}"])
    random.Random(seed).shuffle(rows)
    write_csv('bench_circles.csv', ['x1','x2','target'], rows)

def checkerboard(N=400, seed=42):
    """4x4 checkerboard pattern — requires x1*x2 threshold interaction."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        x1 = rng.uniform(-2, 2)
        x2 = rng.uniform(-2, 2)
        cls = 1 if (math.floor(x1) + math.floor(x2)) % 2 == 0 else 0
        rows.append([f"{x1:.4f}", f"{x2:.4f}", f"{cls}"])
    random.Random(seed).shuffle(rows)
    write_csv('bench_checkerboard.csv', ['x1','x2','target'], rows)

def sine_product_3d(N=400, seed=42):
    """y = sin(x1*x2*x3) — requires 3-way compound interaction."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        x1 = rng.uniform(-2, 2)
        x2 = rng.uniform(-2, 2)
        x3 = rng.uniform(-2, 2)
        y = math.sin(x1 * x2 * x3)
        rows.append([f"{x1:.4f}", f"{x2:.4f}", f"{x3:.4f}", f"{y:.4f}"])
    random.Random(seed).shuffle(rows)
    write_csv('bench_sine3d.csv', ['x1','x2','x3','target'], rows)

def mixed_polynomial(N=400, seed=42):
    """y = x1² + x2² - 0.5*x1*x2*x3 + sin(2*x4) — multiple feature types."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        x1 = rng.uniform(-2, 2)
        x2 = rng.uniform(-2, 2)
        x3 = rng.uniform(-1, 1)
        x4 = rng.uniform(-3, 3)
        y = x1**2 + x2**2 - 0.5*x1*x2*x3 + math.sin(2*x4)
        rows.append([f"{x1:.4f}", f"{x2:.4f}", f"{x3:.4f}", f"{x4:.4f}", f"{y:.4f}"])
    random.Random(seed).shuffle(rows)
    write_csv('bench_mixed_poly.csv', ['x1','x2','x3','x4','target'], rows)

def xor_5d(N=500, seed=42):
    """5-dimensional XOR — parity of 5 binary inputs. Tests high-dim boolean."""
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        bits = [rng.randint(0,1) for _ in range(5)]
        parity = sum(bits) % 2
        rows.append([str(b) for b in bits] + [str(parity)])
    random.Random(seed).shuffle(rows)
    write_csv('bench_xor5d.csv', ['x0','x1','x2','x3','x4','target'], rows)

if __name__ == '__main__':
    outdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(outdir)
    print("Preparing hard benchmark datasets...")
    spiral()
    circles()
    checkerboard()
    sine_product_3d()
    mixed_polynomial()
    xor_5d()
    print("Done!")

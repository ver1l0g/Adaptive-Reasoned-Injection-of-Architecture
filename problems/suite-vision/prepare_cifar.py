#!/usr/bin/env python3
"""Prepare CIFAR-10 for GP-NN. Downloads the archive if needed and
unpickles with pure Python (no TF/torch required).

Outputs:
  bench_cifar_gray_5k.csv   — grayscale 32x32 (1024 inputs), 5000 train samples
  bench_cifar_rgb_5k.csv    — RGB 32x32x3 (3072 inputs), 5000 train samples
One-hot 10-class output. Values kept in 0-255 scale (engine auto-normalizes).
"""
import csv, os, pickle, sys, urllib.request, random

ARCHIVE = "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXTRACT = "cifar-10-batches-py"

def ensure_data():
    if os.path.isdir(EXTRACT):
        return
    if not os.path.isfile(ARCHIVE):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    import tarfile
    with tarfile.open(ARCHIVE) as tf:
        tf.extractall()
    print("Extracted.")

def load_batches(n_batches):
    batches = []
    for i in range(1, n_batches + 1):
        p = os.path.join(EXTRACT, f"data_batch_{i}")
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        batches.append(d)
    return batches

def gray(x):
    # ITU-R BT.601 luma
    r = x[0:1024]; g = x[1024:2048]; b = x[2048:3072]
    return [0.299*ri + 0.587*gi + 0.114*bi for ri, gi, bi in zip(r, g, b)]

def write_csv(fname, rows_gen, n):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        # header length written after first row known — do two-pass simple:
    # simpler: write header first based on cols param via generator design
    raise SystemExit("internal")

def main(n_samples=5000, seed=7):
    ensure_data()
    batches = load_batches(5)  # 5 x 10000 train batches
    X, Y = [], []
    for b in batches:
        X.extend(b[b"data"])
        Y.extend(b[b"labels"])
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    idx = idx[:n_samples]

    # Grayscale version (1024 inputs)
    header = [f"p{i}" for i in range(1024)] + [f"c{i}" for i in range(10)]
    with open("bench_cifar_gray_5k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in idx:
            g = gray(X[i])
            row = [f"{v:.1f}" for v in g] + [0]*10
            row[1024 + Y[i]] = 1
            w.writerow(row)
    print("Wrote bench_cifar_gray_5k.csv")

    # RGB version (3072 inputs)
    header = [f"p{i}" for i in range(3072)] + [f"c{i}" for i in range(10)]
    with open("bench_cifar_rgb_5k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in idx:
            row = [f"{v:.1f}" for v in X[i]] + [0]*10
            row[3072 + Y[i]] = 1
            w.writerow(row)
    print("Wrote bench_cifar_rgb_5k.csv")

if __name__ == "__main__":
    main()

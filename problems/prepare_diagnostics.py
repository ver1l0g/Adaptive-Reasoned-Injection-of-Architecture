#!/usr/bin/env python3
"""Diagnostics for the architectural-limit decision tests.

1. shakespeare_w1: single-char context (pure bigram task). If the engine
   can't model THIS well, the gap is optimization on one-hot data, not
   long-range context mixing.
2. cifar_gray_5k_pooled8: 32x32 gray -> 8x8 average-pooled (64 inputs).
   Hand-feeds the pooling that PATCH_POOLING tries to discover. Compares
   "can't discover pooling" vs "can't use pooled features".
"""
import csv, random

def shakespeare_w1(n=60000, seed=7):
    text = open("tinyshakespeare.txt", encoding="utf-8").read()
    vocab = sorted(set(text))
    V = len(vocab)
    c2i = {c: i for i, c in enumerate(vocab)}
    rng = random.Random(seed)
    with open("bench_shakespeare_w1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["c0"] + [f"t{i}" for i in range(V)])
        for _ in range(n):
            s = rng.randrange(0, len(text) - 1)
            row = [c2i[text[s]]] + [0] * V
            row[1 + c2i[text[s + 1]]] = 1
            w.writerow(row)
    print(f"wrote bench_shakespeare_w1.csv ({n} samples, V={V})")
    # bigram floor for reference
    import math, collections
    from collections import defaultdict
    big = defaultdict(collections.Counter)
    for a, b in zip(text, text[1:]):
        big[a][b] += 1
    N = sum(sum(c.values()) for c in big.values())
    ce = 0.0
    for a, cnt in big.items():
        tot = sum(cnt.values())
        for b, n2 in cnt.items():
            ce -= (n2 / N) * math.log(n2 / tot)
    print(f"bigram floor: {ce:.4f} nats = {ce/0.693147:.3f} bits/char")

def cifar_pooled8(src="bench_cifar_gray_5k.csv", out="bench_cifar_gray_5k_pooled8.csv"):
    import io
    with open(src, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"p{i}" for i in range(64)] + header[1024:])
        for row in rows:
            px = [float(v) for v in row[:1024]]
            pooled = []
            for by in range(8):
                for bx in range(8):
                    s = 0.0
                    for dy in range(4):
                        for dx in range(4):
                            s += px[(by*4+dy)*32 + (bx*4+dx)]
                    pooled.append(s / 16.0)
            w.writerow([f"{v:.1f}" for v in pooled] + row[1024:])
    print(f"wrote {out} ({len(rows)} samples)")

if __name__ == "__main__":
    shakespeare_w1()
    cifar_pooled8()

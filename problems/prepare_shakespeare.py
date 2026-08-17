#!/usr/bin/env python3
"""Char-level language modeling benchmark for GP-NN (tiny-Shakespeare).

Downloads tiny-shakespeare (~1.1MB), builds sliding-window samples:
  inputs:  WINDOW chars (one-hot, vocab-size V)
  target:  next char (one-hot)

Window size is a train-time parameter: LMs live and die by context length,
so three variants test the memory spectrum:
  shakespeare_w8.csv    8-char context    (~60k samples)
  shakespeare_w16.csv   16-char context   (~60k samples)
  shakespeare_w32.csv   32-char context   (~60k samples)

Also writes shakespeare_bpc.py helper values (vocab, counts) to stdout.
Rows are SHUFFLED (context windows are independent samples).
"""
import csv, os, random, urllib.request, collections

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def main(window_list=(8, 16, 32), n_samples=60000, seed=7):
    fname = "tinyshakespeare.txt"
    if not os.path.exists(fname):
        print("Downloading tiny-shakespeare...")
        urllib.request.urlretrieve(URL, fname)
    text = open(fname, encoding="utf-8").read()
    vocab = sorted(set(text))
    V = len(vocab)
    c2i = {c: i for i, c in enumerate(vocab)}
    print(f"corpus: {len(text)} chars, vocab {V}: {''.join(vocab)!r}")

    # unigram bits-per-char baseline (the floor a LM must beat)
    counts = collections.Counter(text)
    import math
    total = sum(counts.values())
    unigram_bpc = -sum((n/total) * math.log2(n/total) for n in counts.values())
    print(f"unigram baseline: {unigram_bpc:.3f} bits/char")

    rng = random.Random(seed)
    for W in window_list:
        out = f"bench_shakespeare_w{W}.csv"
        starts = [rng.randrange(0, len(text) - W - 1) for _ in range(n_samples)]
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            header = [f"c{i}" for i in range(W)] + [f"t{i}" for i in range(V)]
            w.writerow(header)
            for s in starts:
                row = []
                for k in range(W):
                    row.append(c2i[text[s + k]])
                row.extend([0] * V)
                row[W + c2i[text[s + W]]] = 1
                w.writerow(row)
        print(f"wrote {out}: {n_samples} samples, {W}+{V} cols")
    print(f"VOCAB={V}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stripes20 shape analysis: run widths, levels, sample density per stripe."""
import csv

f = csv.reader(open("suite-limits/bench_stripes20.csv"))
next(f)
rows = [(float(r[0]), float(r[1])) for r in f]
print("n =", len(rows), "| x range:",
      round(min(x for x, y in rows), 3), round(max(x for x, y in rows), 3))
print("y levels:", sorted(set(round(y, 2) for x, y in rows)))

runs = []
prev = None
start = None
for x, y in sorted(rows) + [(999.0, None)]:
    if y != prev:
        if prev is not None:
            runs.append((start, x, prev))
        start = x
        prev = y
print("stripe runs:", len(runs))
print("first 6:", [(round(a, 2), round(b, 2), c) for a, b, c in runs[:6]])
widths = [b - a for a, b, c in runs]
counts = [sum(1 for x, y in rows if a <= x < b) for a, b, c in runs]
print("min/median run width:", round(min(widths), 3), round(sorted(widths)[len(widths)//2], 3))
print("min/median samples per run:", min(counts), sorted(counts)[len(counts)//2])

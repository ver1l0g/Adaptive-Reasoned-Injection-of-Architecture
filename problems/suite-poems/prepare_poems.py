#!/usr/bin/env python3
"""suite-poems prep: extract 五言绝句 (5-char quatrains) from Quan-Tangshi
and emit ARIA-format next-char CSVs with a strict-structure test bed.

Task shape (mirrors suite-language): W=19 context slots (chars 1..19 of a
20-char poem, PAD=0 start code), V outputs one-hot (top-1000 hanzi, UNK).
The structure signal: char 5/10/15/20 are line-final (rhyme/parallelism
constraints); position 1 of lines 2-4 follow couplet structure. A model
that DISCOVERS this without being told = structure induction, the ARIA
thesis test. Also writes position-conditional unigram stats for baselines.

Usage: python prepare_poems.py   (from suite-poems/)
"""
import glob
import json
import math
import os
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

RAW = os.path.join("raw", "cprepo")
TANG = "\u5168\u5510\u8bd7"          # 全唐诗
PUNCT = "\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a\u201c\u201d\u300a\u300b\uff08\uff09\u2014\u2026\u3000"
PAD = 0
V = 501          # top-500 hanzi + UNK(500); PAD=0 -> 501 output cols
W = 19
MAX_ROWS = 15000


def clean(s):
    out = []
    for ch in s:
        if ch in PUNCT:
            continue
        out.append(ch)
    return "".join(out)


def is_wujue(entry):
    """五言绝句: 2 paragraph strings, each '5,5。' — 4 lines x 5 chars."""
    ps = entry.get("paragraphs") or []
    if len(ps) != 2:
        return None
    lines = []
    for p in ps:
        # split on ，and 。 → two 5-char lines each
        parts = [clean(x) for x in p.replace("\uff0c", "\uff0c").split("\uff0c")]
        parts = [x for x in parts if x]
        # after cleaning punctuation: '5,5。' -> two strings of len 5
        if len(parts) != 2 or any(len(x) != 5 for x in parts):
            return None
        lines.extend(parts)
    if len(lines) != 4:
        return None
    return lines  # 4 strings of 5 hanzi


def main():
    files = sorted(glob.glob(os.path.join(RAW, TANG, "poet.tang.*.json")))
    print(f"source files: {len(files)}")
    poems = []
    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for e in data:
            lines = is_wujue(e)
            if lines:
                poems.append("".join(lines))
    print(f"五言绝句 extracted: {len(poems)}")

    # vocab over the filtered set
    cnt = Counter()
    for p in poems:
        cnt.update(p)
    top = [ch for ch, _ in cnt.most_common(V - 1)]
    vocab = {ch: i + 1 for i, ch in enumerate(top)}   # 1..500; 0=PAD, 500=UNK
    unk = V - 1
    unk_rate = 1.0 - sum(c for ch, c in cnt.items() if ch in vocab) / max(1, sum(cnt.values()))
    print(f"vocab: {len(vocab)} + PAD/UNK | UNK rate: {unk_rate:.4f}")
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for ch, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(f"{i}\t{ch}\n")

    def code(ch):
        return vocab.get(ch, unk)

    # rows: for t in 1..19: window = codes of chars[0..t-1] right-aligned,
    # left-padded with PAD; target = char t. (Position identity is implicit
    # in the padding — the model sees how far into the poem it is.)
    rows = []
    for p in poems:
        cs = [code(c) for c in p]           # 20 codes
        for t in range(1, 20):
            ctx = cs[:t][-W:]
            ctx = [PAD] * (W - len(ctx)) + ctx
            target = cs[t]
            rows.append((ctx, t, target))
    random.Random(42).shuffle(rows)
    if len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS]     # overnight-viable: the 1001-col config
                                    # cost ~2h/epoch (59k rows x 19k weights)
    n_train = int(len(rows) * 0.8)

    def write_csv(path, subset):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join([f"in_{i}" for i in range(W)])
                    + "," + ",".join([f"out_{j}" for j in range(V)]) + "\n")
            for ctx, t, tgt in subset:
                onehot = ["0"] * V
                onehot[tgt] = "1"
                f.write(",".join(str(c) for c in ctx) + "," + ",".join(onehot) + "\n")

    write_csv("bench_wujue_train.csv", rows[:n_train])
    write_csv("bench_wujue_test.csv", rows[n_train:])
    print(f"rows: {len(rows)} total | train {n_train} / test {len(rows) - n_train}")

    # Position-conditional unigram stats (structure baseline for analysis):
    with open("position_stats.txt", "w", encoding="utf-8") as f:
        for pos in range(20):
            c = Counter(p[pos] for p in poems if len(p) == 20)
            ent = 0.0
            tot = sum(c.values())
            for v in c.values():
                p_ = v / tot
                ent -= p_ * (p_ and __import__("math").log2(p_))
            top3 = ", ".join(f"{ch}({v})" for ch, v in c.most_common(3))
            f.write(f"pos {pos:2d}: entropy {ent:6.3f} bits | top3 {top3}\n")
    print("position_stats.txt written (line-position structure baselines)")


if __name__ == "__main__":
    main()

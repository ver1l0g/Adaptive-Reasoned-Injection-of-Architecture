#!/usr/bin/env python3
"""Prepare CIFAR-10 from fast.ai PNG mirror for GP-NN.

Reads cifar10/train/<class>/*.png, writes:
  bench_cifar_gray_5k.csv  — grayscale 32x32 (1024 inputs), ~5000 balanced samples
  bench_cifar_rgb_5k.csv   — RGB 3072 inputs, same samples
Values 0-255 (engine auto-normalizes). One-hot 10 output columns.
"""
import csv, os, random
from PIL import Image

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
PER_CLASS = 500  # 5000 total

def main(seed=7):
    import sys
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    root = "cifar10"
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"{train_dir} not found — extract cifar10_fastai.tgz first")

    rng = random.Random(seed)
    samples = []  # (path, class_idx)
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(train_dir, cls)
        files = sorted(os.listdir(d))
        rng.shuffle(files)
        for fn in files[:per_class]:
            samples.append((os.path.join(d, fn), ci))
    rng.shuffle(samples)
    print(f"{len(samples)} samples")

    out_gray = f"bench_cifar_gray_{len(samples)//1000}k.csv"
    fg = open(out_gray, "w", newline="")
    wg = csv.writer(fg)
    wg.writerow([f"p{i}" for i in range(1024)] + [f"c{i}" for i in range(10)])

    for path, ci in samples:
        img = Image.open(path).convert("RGB")
        px = list(img.getdata())  # [(r,g,b), ...] 1024 pixels
        gray = [0.299*r + 0.587*g + 0.114*b for (r, g, b) in px]
        rowg = [f"{v:.1f}" for v in gray] + [0]*10
        rowg[1024 + ci] = 1
        wg.writerow(rowg)

    fg.close()
    print(f"Wrote {out_gray}")

if __name__ == "__main__":
    main()

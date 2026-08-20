import os, random, sys
from PIL import Image

files = []
for r, ds, fs in os.walk('cifar10/train'):
    files.extend(os.path.join(r, f) for f in fs)
random.seed(1)
bad = 0
for p in random.sample(files, 300):
    try:
        Image.open(p).verify()
    except Exception:
        bad += 1
        print("corrupt:", p)
print("corrupt in sample:", bad, "/", 300)

#!/usr/bin/env python3
"""M5.2 probe: I.32.8 commit forensics 鈥?where does the 0.983 plateau come from?"""
import re, subprocess, sys, time

EXE = r"..\aria26.exe"
R2A_RE = re.compile(r"Eval R2\[out(\d+)\]:\s*([0-9.eE+-]+)")
COMMIT_RE = re.compile(r"COMMIT rank=\d+ type=(\w+) val_loss=([\d.eE+-]+)")
PROMISE_RE = re.compile(r"DIVIDE_PRODUCT \(input=(\S+)\)")

wd = "i328probe"
import os
os.makedirs(wd, exist_ok=True)
train = os.path.abspath(r"..\problems\suite-feynman\I.32.8.csv")
test = os.path.abspath(r"..\problems\suite-feynman\I.32.8_test.csv")
args = [EXE, "--csv", train, "--input-cols", "3", "--eval-csv", test,
        "--max-epochs", "150", "--seed", "1", "--save-graph", "none"]
log = os.path.join(wd, "log.txt")
t0 = time.time()
with open(log, "w", encoding="utf-8", errors="replace") as lf:
    subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
text = open(log, encoding="utf-8", errors="replace").read()
print("commits:")
for m in COMMIT_RE.finditer(text):
    print(f"  {m.group(1)} val={m.group(2)}")
r2 = R2A_RE.search(text)
print(f"final: R2={r2.group(2) if r2 else 'n/a'} ({time.time()-t0:.0f}s)")
divs = PROMISE_RE.findall(text)
print(f"DIVIDE_PRODUCT emissions: {len(divs)} (unique: {len(set(divs))})")


#!/usr/bin/env python3
"""Crash discriminator: aria11/aria12 on charLM w8 under library states.
A = fresh dir (no libraries), B = library only, C = library + failure lib.
Captures Windows exit codes (0xC0000005 = access violation, -1073741510 = CTRL close).
"""
import os, shutil, subprocess, sys, time

EXE = os.path.abspath(sys.argv[1])
MODE = sys.argv[2]   # A | B | C
wd = os.path.join("crashprobe", MODE)
os.makedirs(wd, exist_ok=True)
if MODE in ("B", "C"):
    shutil.copy2("subgraph_library.txt", os.path.join(wd, "subgraph_library.txt"))
if MODE == "C":
    shutil.copy2("failure_library.txt", os.path.join(wd, "failure_library.txt"))

csv_path = os.path.abspath("suite-language/bench_shakespeare_w8.csv")
args = [EXE, "--csv", csv_path, "--input-cols", "8", "--output-cols", "65",
        "--loss", "bce", "--max-epochs", "12", "--seed", "1",
        "--save-graph", "none", "--eval-csv", csv_path]
t0 = time.time()
with open(os.path.join(wd, "run_log.txt"), "w", encoding="utf-8",
          errors="replace") as lf:
    p = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, cwd=wd,
                       timeout=1800)
print(f"mode={MODE} exe={os.path.basename(EXE)} rc={p.returncode} "
      f"({time.time()-t0:.0f}s)", flush=True)

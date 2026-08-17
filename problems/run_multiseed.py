#!/usr/bin/env python3
"""Multi-seed benchmark harness for GP-NN.

Runs each task across N seeds and reports per-task final-loss statistics
(mean / std / min / max), so a change can be judged against the engine's
run-to-run RNG noise instead of a single shot. The engine is reproducible
per seed (--seed S), so these numbers are stable across harness runs.

Usage:
    python run_multiseed.py                       # all 17 tasks, 5 seeds, 200 ep
    python run_multiseed.py --seeds 3 --epochs 100
    python run_multiseed.py --tasks d4 d5 t31     # subset
    python run_multiseed.py --jobs 1              # sequential (accurate timing)

Final-loss numbers go to stdout (table) and to multiseed_results.csv.

NOTE: the engine uses ~20 internal threads, so running many gpnn.exe
concurrently oversubscribes. Loss statistics are unaffected; only wall-time
is. Use --jobs 1 for honest per-task timing.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Task registry: name -> (csv, input-cols, [extra args])
TASKS = {
    "t21": ("bench_t21_three_region.csv",       1, []),
    "t22": ("bench_t22_windowed_sine.csv",      1, []),
    "t23": ("bench_t23_quadratic.csv",          2, []),
    "t24": ("bench_t24_quadrant_xor.csv",       2, []),
    "t31": ("bench_t31_three_way_product.csv",  3, []),
    "t32": ("bench_t32_sine_of_product.csv",    2, []),
    "t33": ("bench_t33_absolute_value.csv",     1, []),
    "t34": ("bench_t34_xor3_parity.csv",        3, []),
    "d1":  ("bench_d1_step.csv",                1, []),
    "d2":  ("bench_d2_mod3.csv",                1, []),
    "d3":  ("bench_d3_two_sines.csv",           1, []),
    "d4":  ("bench_d4_irrelevant12.csv",       12, []),
    "d5":  ("bench_d5_extrap_cliff.csv",        1, ["--sweep", "1.0", "3.0", "0.25"]),
    "d6":  ("bench_d6_seq_parity.csv",          1, ["--no-shuffle"]),
    "d7":  ("bench_d7_multiout.csv",            1, ["--output-cols", "3"]),
    "d8":  ("bench_d8_compose_absprod.csv",     2, []),
    "d9":  ("bench_d9_noise.csv",               2, []),
}

FINAL_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")
STRUCT_RE = re.compile(r"Structural changes:\s*(\d+)")
WALL_RE = re.compile(r"Wall time:\s*(\d+)\s*ms")


def run_one(name, csvf, incols, extra, epochs, seed, exe):
    args = [exe, "--csv", csvf, "--input-cols", str(incols),
            "--max-epochs", str(epochs), "--seed", str(seed)] + extra
    # --log-file NUL silences the per-run log file (Windows); harmless elsewhere.
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return name, seed, None, None, None, "TIMEOUT"
    out = proc.stdout or ""
    # Send log to a discard file via a second pass is awkward; instead we pass
    # --log-file to a temp path below if needed. For now the engine writes a
    # default log only if --log-file is given, so we're fine.
    fm = FINAL_RE.search(out)
    sm = STRUCT_RE.search(out)
    wm = WALL_RE.search(out)
    if not fm:
        return name, seed, None, None, None, (proc.stderr.strip()[:120] or "no Final loss line")
    final = float(fm.group(1))
    struct = int(sm.group(1)) if sm else None
    wall = int(wm.group(1)) if wm else None
    return name, seed, final, struct, wall, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (default 5)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent gpnn.exe runs")
    ap.add_argument("--exe", default="../gpnn.exe")
    args = ap.parse_args()

    tasks = args.tasks if args.tasks else list(TASKS.keys())
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        sys.exit(f"unknown tasks: {bad}; known: {list(TASKS)}")

    jobs = []
    for name in tasks:
        csvf, incols, extra = TASKS[name]
        for s in range(1, args.seeds + 1):
            jobs.append((name, csvf, incols, extra, s))

    print(f"Running {len(jobs)} runs ({len(tasks)} tasks x {args.seeds} seeds), "
          f"{args.jobs} concurrent...")

    results = {}  # name -> list of (seed, final, struct, wall, err)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, n, c, ic, ex_, args.epochs, s, args.exe)
                for (n, c, ic, ex_, s) in jobs]
        done = 0
        for f in as_completed(futs):
            name, seed, final, struct, wall, err = f.result()
            results.setdefault(name, []).append((seed, final, struct, wall, err))
            done += 1
            tag = f"{name} seed{seed}: " + (f"final={final:.6g}" if err is None else f"ERR={err}")
            print(f"  [{done}/{len(jobs)}] {tag}")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'task':<5} {'n':>3} {'mean':>13} {'std':>11} {'min':>13} {'max':>13}  errs")
    print("-" * 78)
    rows = []
    for name in sorted(results):
        rs = results[name]
        ok = [r[1] for r in rs if r[4] is None and r[1] is not None]
        errs = len(rs) - len(ok)
        if ok:
            mean = statistics.mean(ok)
            std = statistics.pstdev(ok) if len(ok) > 1 else 0.0
            mn, mx = min(ok), max(ok)
            print(f"{name:<5} {len(ok):>3} {mean:>13.6g} {std:>11.6g} "
                  f"{mn:>13.6g} {mx:>13.6g}  {errs}")
            rows.append((name, len(ok), mean, std, mn, mx))
        else:
            print(f"{name:<5} {0:>3}  (all failed)")
            rows.append((name, 0, None, None, None, None))

    with open("multiseed_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_ok", "mean", "std", "min", "max"])
        for r in rows:
            w.writerow([r[0], r[1]] + ["" if v is None else f"{v:.9g}" for v in r[2:]])
    print(f"\nWrote multiseed_results.csv")


if __name__ == "__main__":
    main()

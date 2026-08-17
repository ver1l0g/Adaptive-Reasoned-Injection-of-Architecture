#!/usr/bin/env python3
"""Char-LM harness: window 8 -> 16 -> 32, reports bits/char.

BCE (natural log, summed over 65 outputs) / ln(2) = bits per char.
Reference: unigram 4.78, char-RNN ~1.6, good char-LM ~1.4 on this corpus.
Window CSVs are shuffled — no --no-shuffle.

PARSE-BUG FIX: subprocess stdout is now tee'd to a per-run log file
(lm_<task>_run.txt) BEFORE parsing, so results survive harness crashes,
timeouts, and process kills. The regexes parse the FILE (always complete
on disk) rather than the in-memory string (lost on TimeoutExpired, and
empty when the child was killed). Fallback parsing from the log file also
covers the case where the summary line was never printed.
"""
import re, subprocess, sys, time, os

EXE = sys.argv[1] if len(sys.argv) > 1 else "../gpnn7.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
V = 65
LN2 = 0.6931471805599453

TASKS = [
    ("w8",  8,  "bench_shakespeare_w8.csv"),
    ("w16", 16, "bench_shakespeare_w16.csv"),
    ("w32", 32, "bench_shakespeare_w32.csv"),
]

VAL_RE  = re.compile(r"Restored best-val graph snapshot \(val=([0-9.eE+-]+)")
LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")
COM_RE  = re.compile(r"COMMIT rank=\d+ type=(\w+)")
# Fallbacks for killed runs (no summary lines): last epoch loss + best val seen
EPOCH_RE = re.compile(r"sgd\s+ loss=([0-9.eE+-]+)")
BESTVAL_RE = re.compile(r"New best val loss: ([0-9.eE+-]+)")

def parse_run(logpath):
    """Parse results from a run log file. Returns (val, loss, commits)."""
    try:
        text = open(logpath, encoding="utf-8", errors="replace").read()
    except OSError:
        return None, None, "no-log"
    from collections import Counter
    commits = ",".join(f"{k}x{v}" for k, v in Counter(COM_RE.findall(text)).items()) or "none"
    vm = VAL_RE.search(text)
    val = float(vm.group(1)) if vm else None
    if val is None:
        # fallback: best "New best val loss" line
        bm = BESTVAL_RE.findall(text)
        val = float(bm[-1]) if bm else None
    lm = LOSS_RE.search(text)
    loss = float(lm.group(1)) if lm else None
    if loss is None:
        em = EPOCH_RE.findall(text)
        loss = float(em[-1]) if em else None
    return val, loss, commits

def main():
    for name, win, csvf in TASKS:
        t0 = time.time()
        logpath = f"lm_{name}_run.txt"
        full = [EXE, "--csv", csvf, "--input-cols", str(win),
                "--output-cols", str(V), "--loss", "bce",
                "--max-epochs", str(EPOCHS), "--seed", "1",
                "--save-graph", "none"]
        # Tee: run with stdout redirected to the log file by the shell
        # wrapper, so bytes hit disk as they are produced.
        try:
            with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                proc = subprocess.run(full, stdout=lf, stderr=subprocess.STDOUT,
                                      timeout=14400)
        except subprocess.TimeoutExpired:
            pass  # log file still has everything printed so far
        val, loss, commits = parse_run(logpath)
        val_bpc = f"{val/LN2:.3f}" if val is not None else "n/a"
        tr_bpc = f"{loss/LN2:.3f}" if loss is not None else "n/a"
        print(f"{name:<5} val={val_bpc} bits/char  (train {tr_bpc})  "
              f"commits={commits}  ({(time.time()-t0)/60:.0f}min)", flush=True)

if __name__ == "__main__":
    main()

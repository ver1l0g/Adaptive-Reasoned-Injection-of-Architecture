#!/usr/bin/env python3
"""Feynman subset harness: run gpnn on all 25 equations, report test R虏.

Sequential (one equation at a time) 鈥?designed to coexist with other
background chains. Writes results incrementally to results/feynman_results.csv
so progress survives interruption. SRBench-style thresholds:
  R虏 > 0.9999 = exact recovery, R虏 > 0.99 = solved.
"""
import csv, os, re, subprocess, sys, time

EXE = sys.argv[1] if len(sys.argv) > 1 else "../gpnn4.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1

EQS = [
    ("I.6.2",2),("I.6.2b",2),("I.7.9",2),("I.8.4",2),("I.9.5",3),
    ("I.10.7",2),("I.12.5",3),("I.13.12",2),("I.14.3",4),("I.15.3",2),
    ("I.15.10",2),("I.18.4",3),("I.18.14",3),("I.24.6",2),("I.25.9",2),
    ("I.29.16",3),("I.32.8",3),("I.34.6",1),("I.43.27",3),("I.47.23",2),
    ("II.11.27",3),("II.34.29a",2),("III.4.33",2),("III.10.19",2),("I.48.20",3),
]

R2_RE = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")
LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")
EXPR_RE = re.compile(r"\[Expression\]\s*(.*)")

def main():
    results = []
    start = time.time()
    out_path = f"results/feynman_results_seed{SEED}.csv" if SEED != 1 else "results/feynman_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eq", "vars", "test_r2", "train_loss", "wall_s", "expression"])
        for i, (eid, nvars) in enumerate(EQS):
            t0 = time.time()
            logpath = f"feynman_run_{eid}.txt"
            args = [EXE, "--csv", f"suite-feynman/{eid}.csv",
                    "--input-cols", str(nvars),
                    "--eval-csv", f"suite-feynman/{eid}_test.csv",
                    "--max-epochs", str(EPOCHS), "--seed", str(SEED)]
            # Tee stdout to a per-equation log (crash-safe: parse from disk)
            try:
                with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                    subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=2700)
            except subprocess.TimeoutExpired:
                pass
            try:
                out = open(logpath, encoding="utf-8", errors="replace").read()
            except OSError:
                out = ""
            r2m = R2_RE.search(out)
            lm = LOSS_RE.search(out)
            em = EXPR_RE.search(out)
            r2 = float(r2m.group(1)) if r2m else None
            loss = float(lm.group(1)) if lm else None
            expr = (em.group(1).strip()[:120] if em else "")
            wall = round(time.time() - t0, 1)
            results.append(r2)
            w.writerow([eid, nvars, "" if r2 is None else f"{r2:.6f}",
                        "" if loss is None else f"{loss:.6g}", wall, expr])
            f.flush()
            tag = f"R2={r2:.4f}" if r2 is not None else "FAIL"
            print(f"[{i+1}/{len(EQS)}] {eid:<10} {tag}  ({wall}s)", flush=True)

    ok = [r for r in results if r is not None]
    exact = sum(1 for r in ok if r > 0.9999)
    solved = sum(1 for r in ok if r > 0.99)
    print(f"\n=== Feynman subset: {len(EQS)} equations, {EPOCHS} epochs, seed {SEED} ===")
    print(f"exact (R2>0.9999): {exact}/{len(EQS)}   solved (R2>0.99): {solved}/{len(EQS)}")
    print(f"total wall: {round((time.time()-start)/60,1)} min")

if __name__ == "__main__":
    main()

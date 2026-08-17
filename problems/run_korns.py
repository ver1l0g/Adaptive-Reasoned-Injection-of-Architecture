#!/usr/bin/env python3
"""Korns harness: 9 problems, test R2 per problem. Sequential; light."""
import csv, re, subprocess, sys, time

EXE = sys.argv[1] if len(sys.argv) > 1 else "../gpnn5.exe"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

R2_RE = re.compile(r"Eval R2\[out0\]:\s*([0-9.eE+-]+)")
LOSS_RE = re.compile(r"Final loss:\s*([0-9.eE+-]+)")

def main():
    results = []
    with open("korns_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prob", "test_r2", "train_loss", "wall_s"])
        for i, fid in enumerate(["F1","F2","F3","F4","F5","F6","F7","F8","F9"]):
            t0 = time.time()
            args = [EXE, "--csv", f"korns/{fid}.csv", "--input-cols", "5",
                    "--eval-csv", f"korns/{fid}_test.csv",
                    "--max-epochs", str(EPOCHS), "--seed", "1"]
            # Tee stdout to a per-problem log (crash-safe: parse from disk)
            logpath = f"korns_run_{fid}.txt"
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
            r2m = R2_RE.search(out); lm = LOSS_RE.search(out)
            r2 = float(r2m.group(1)) if r2m else None
            loss = float(lm.group(1)) if lm else None
            wall = round(time.time() - t0, 1)
            results.append(r2)
            w.writerow([fid, "" if r2 is None else f"{r2:.6f}",
                        "" if loss is None else f"{loss:.6g}", wall])
            f.flush()
            tag = f"R2={r2:.4f}" if r2 is not None else "FAIL"
            print(f"[{i+1}/9] {fid}: {tag}  ({wall}s)", flush=True)

    ok = [r for r in results if r is not None]
    solved = sum(1 for r in ok if r > 0.99)
    print(f"\n=== Korns: solved (R2>0.99): {solved}/9 ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M6.1 Baseline harness: MLP (sklearn) + optional PySR on ARIA's suites.

Head-to-head comparison with IDENTICAL splits (strided 1000/500 like ARIA
uses: first 2/3 train, last 1/3 test — matching Dataset::split's
no-shuffle mode for feynman/korns which are pre-shuffled files).

Outputs baselines_results.csv: task, model, test_r2, wall_s.
"""
import csv, math, os, re, subprocess, sys, time

def load_xy(path, n_in):
    X, y = [], []
    with open(path) as f:
        rows = [r for r in csv.reader(f)]
    # auto-header: first row non-numeric
    start = 0
    try:
        float(rows[0][0])
    except ValueError:
        start = 1
    for r in rows[start:]:
        vals = [float(v) for v in r]
        X.append(vals[:n_in])
        y.append(vals[n_in])
    return X, y

def r2_score(y_true, y_pred):
    n = len(y_true)
    mean = sum(y_true) / n
    ss_tot = sum((v - mean) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

def run_mlp(X_train, y_train, X_test, y_test, seed=1):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xs = scaler.transform(X_test)
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000,
                     random_state=seed, early_stopping=True)
    m.fit(Xt, y_train)
    pred = m.predict(Xs)
    return r2_score(y_test, pred)

FEYNMAN = [
    ("I.6.2",2),("I.6.2b",2),("I.7.9",2),("I.8.4",2),("I.9.5",3),
    ("I.10.7",2),("I.12.5",3),("I.13.12",2),("I.14.3",4),("I.15.3",2),
    ("I.15.10",2),("I.18.4",3),("I.18.14",3),("I.24.6",2),("I.25.9",2),
    ("I.29.16",3),("I.32.8",3),("I.34.6",1),("I.43.27",3),("I.47.23",2),
    ("II.11.27",3),("II.34.29a",2),("III.4.33",2),("III.10.19",2),("I.48.20",3),
]
KORNS = [("F1",5),("F2",5),("F3",5),("F4",5),("F5",5),
         ("F6",5),("F7",5),("F8",5),("F9",5)]

def main():
    use_pysr = "--pysr" in sys.argv
    suites = []
    for name, nin in FEYNMAN:
        suites.append((f"feynman/{name}", nin))
    for name, nin in KORNS:
        suites.append((f"korns/{name}", nin))

    out_path = "results/baselines_results.csv"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "test_r2", "wall_s"])

        pysr_model = None
        if use_pysr:
            try:
                from pysr import PySRRegressor
                pysr_model = PySRRegressor(
                    niterations=40, binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sin"], populations=8,
                    deterministic=True, random_state=1, verbosity=0,
                    procs=4)
                print("PySR loaded")
            except Exception as e:
                print(f"PySR unavailable ({e}), MLP only")
                use_pysr = False

        for task, nin in suites:
            train = f"suite-{task.split('/')[0]}/{task.split('/')[1]}.csv"
            test  = f"suite-{task.split('/')[0]}/{task.split('/')[1]}_test.csv"
            if not os.path.exists(train):
                continue
            X, y = load_xy(train, nin)
            Xt, yt = load_xy(test, nin)
            # ARIA split: train file has 1000, test file has 500
            # baseline uses them as-is (identical data)

            t0 = time.time()
            mlp_r2 = run_mlp(X, y, Xt, yt)
            w.writerow([task, "MLP(64,64)", f"{mlp_r2:.6f}",
                        round(time.time()-t0, 1)])
            f.flush()
            line = f"{task}: MLP R2={mlp_r2:.4f}"

            if use_pysr and pysr_model is not None:
                t0 = time.time()
                try:
                    import numpy as np
                    pysr_model.fit(np.array(X), np.array(y))
                    pred = pysr_model.predict(np.array(Xt))
                    pysr_r2 = r2_score(yt, list(pred))
                    w.writerow([task, "PySR-40it", f"{pysr_r2:.6f}",
                                round(time.time()-t0, 1)])
                    f.flush()
                    line += f"  PySR R2={pysr_r2:.4f}"
                except Exception as e:
                    w.writerow([task, "PySR-40it", "ERROR: " + str(e)[:60], 0])
                    f.flush()
                    line += f"  PySR ERROR"
            print(line, flush=True)

    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()

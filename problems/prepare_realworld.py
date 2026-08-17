#!/usr/bin/env python3
"""Prepare real-world ML benchmark datasets for GP-NN.

Generates CSV files compatible with the gpnn --csv flag.
"""
import csv
import sys
import os

def try_sklearn():
    try:
        from sklearn.datasets import load_iris, load_breast_cancer, load_wine
        return True
    except ImportError:
        return False

def write_csv(filename, header, rows):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {filename}: {len(rows)} samples, {len(header)-1} columns")

def prepare_iris(outdir):
    """Iris: 4 features, 3 classes (one-hot BCE)."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    # One-hot encode 3 classes
    header = ['f0','f1','f2','f3','c0','c1','c2']
    rows = []
    for i in range(len(X)):
        row = list(X[i]) + [0,0,0]
        row[4 + y[i]] = 1
        rows.append([f"{v:.4f}" for v in row])
    write_csv(os.path.join(outdir, 'bench_iris.csv'), header, rows)

def prepare_breast_cancer(outdir):
    """Breast Cancer Wisconsin: 30 features, binary (BCE)."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    header = [f'f{i}' for i in range(30)] + ['target']
    rows = []
    for i in range(len(X)):
        row = list(X[i]) + [y[i]]
        rows.append([f"{v:.6f}" for v in row])
    write_csv(os.path.join(outdir, 'bench_breast_cancer.csv'), header, rows)

def prepare_wine(outdir):
    """Wine: 13 features, 3 classes (one-hot BCE)."""
    from sklearn.datasets import load_wine
    data = load_wine()
    X, y = data.data, data.target
    header = [f'f{i}' for i in range(13)] + ['c0','c1','c2']
    rows = []
    for i in range(len(X)):
        row = list(X[i]) + [0,0,0]
        row[13 + y[i]] = 1
        rows.append([f"{v:.4f}" for v in row])
    write_csv(os.path.join(outdir, 'bench_wine.csv'), header, rows)

def prepare_iris_mse(outdir):
    """Iris as regression: 4 features, 1 target (class index). MSE."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target
    header = ['f0','f1','f2','f3','target']
    rows = []
    for i in range(len(X)):
        row = list(X[i]) + [y[i]]
        rows.append([f"{v:.4f}" for v in row])
    write_csv(os.path.join(outdir, 'bench_iris_mse.csv'), header, rows)

if __name__ == '__main__':
    outdir = os.path.dirname(os.path.abspath(__file__))
    if not try_sklearn():
        print("scikit-learn not available. Install with: pip install scikit-learn")
        sys.exit(1)
    print("Preparing real-world datasets...")
    prepare_iris(outdir)
    prepare_iris_mse(outdir)
    prepare_breast_cancer(outdir)
    prepare_wine(outdir)
    print("Done!")

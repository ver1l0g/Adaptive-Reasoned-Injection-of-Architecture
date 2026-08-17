"""Prepare 8x8 MNIST digits (sklearn) for GP-NN and run baseline comparison.

Outputs:
  bench_digits.csv     — 1797 samples, 64 inputs (pixel intensities 0-16,
                         scaled to [-1,1]), 10 outputs (one-hot 0/1).
  baseline_results.txt — sklearn LogisticRegression accuracy for comparison.

GP-NN usage:
  cd problems
  ..\gpnn.exe --csv bench_digits.csv --input-cols 64 --output-cols 10 \
      --loss bce --max-epochs 200 --seed 1
"""
import numpy as np
import sys

# ---------------------------------------------------------------------------
# Load sklearn digits (8x8, 1797 samples, 10 classes, pixel values 0-16)
# ---------------------------------------------------------------------------
try:
    from sklearn.datasets import load_digits
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)

digits = load_digits()
X = digits.data.astype(np.float64)   # (1797, 64), range [0, 16]
y = digits.target.astype(int)        # (1797,), range [0, 9]

print(f"Loaded digits: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# ---------------------------------------------------------------------------
# Preprocess: scale pixels to [-1, 1], one-hot encode labels
# ---------------------------------------------------------------------------
# Pixel values are 0-16. Map to [-1, 1]: x_n = (x / 8) - 1
X_norm = (X / 8.0) - 1.0

# One-hot encode: 10 outputs, each 0.0 or 1.0
n_classes = 10
Y_onehot = np.zeros((len(y), n_classes))
for i, label in enumerate(y):
    Y_onehot[i, label] = 1.0

# ---------------------------------------------------------------------------
# Write GP-NN CSV: 64 input columns + 10 output columns = 74 per row
# ---------------------------------------------------------------------------
with open("bench_digits.csv", "w") as f:
    for i in range(len(y)):
        vals = list(X_norm[i]) + list(Y_onehot[i])
        f.write(",".join(f"{v:.4f}" for v in vals) + "\n")

print(f"Wrote bench_digits.csv: {len(y)} samples, 64 inputs, 10 outputs (BCE)")

# Class distribution
for c in range(n_classes):
    count = np.sum(y == c)
    print(f"  digit {c}: {count} samples ({100*count/len(y):.1f}%)")

# ---------------------------------------------------------------------------
# Baseline: sklearn LogisticRegression (one-vs-rest on the same data)
# ---------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

print("\n--- Baseline (5-fold cross-validation) ---")
clf = LogisticRegression(max_iter=1000, solver='lbfgs')
scores = cross_val_score(clf, X_norm, y, cv=5, scoring='accuracy')
print(f"LogisticRegression accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"  (individual folds: {[f'{s:.3f}' for s in scores]})")

# BCE loss of the logistic regression (for comparison with GP-NN's loss)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)

# Compute BCE on the test set (same metric GP-NN uses)
from sklearn.metrics import log_loss
bce = log_loss(y_test, y_proba)
test_acc = clf.score(X_test, y_test)
print(f"\nLogisticRegression on 80/20 split:")
print(f"  Test accuracy: {test_acc:.3f}")
print(f"  Test BCE loss: {bce:.4f}")
print(f"\nGP-NN should target BCE < {bce:.4f} to match logistic regression.")

# Write baseline results
with open("baseline_results.txt", "w") as f:
    f.write(f"LogisticRegression 5-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}\n")
    f.write(f"LogisticRegression 80/20 test accuracy: {test_acc:.3f}\n")
    f.write(f"LogisticRegression 80/20 test BCE: {bce:.4f}\n")
    f.write(f"\nGP-NN target: BCE < {bce:.4f}\n")

print(f"\nWrote baseline_results.txt")
print(f"\nTo run GP-NN:")
print(f"  cd problems")
print(f"  ..\\gpnn.exe --csv bench_digits.csv --input-cols 64 --output-cols 10 --loss bce --max-epochs 200 --seed 1")

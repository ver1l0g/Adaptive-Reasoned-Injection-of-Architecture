"""Generate multi-component Fourier time series for GP-NN.

Signal: y(t) = sin(0.3t) + 0.5*sin(0.7t) + 0.3*sin(1.3t)

Three non-commensurate frequencies (no integer ratios between them), so the
combined signal has a very long repeat period. The challenge for a recurrent
predictor: the mapping y(t-1) -> y(t) is MULTI-VALUED — the same y(t-1) can
be produced by different phase combinations of the three sinusoids, leading to
different y(t). A single recurrent state (1-D) can encode at most one phase;
this signal needs three. The question: can the structural search discover it
needs MULTIPLE recurrent connections, one per frequency component?

Datasets:
  bench_fourier_ff.csv  — input = t (normalized), output = y(t). Function
                          fitting. Tests Fourier decomposition directly.
  bench_fourier_ts1.csv — input = y(t-1), output = y(t). Time series with
                          --no-shuffle. Tests recurrence's multi-phase capacity.
  bench_fourier_ts3.csv — input = [y(t-1), y(t-5), y(t-10)], output = y(t).
                          Embedded approach (no recurrence needed).
"""
import math

T = 1000  # total time steps
y = []
for t in range(T):
    val = math.sin(0.3*t) + 0.5*math.sin(0.7*t) + 0.3*math.sin(1.3*t)
    y.append(val)

var_y = sum(v*v for v in y) / len(y) - (sum(y)/len(y))**2
print(f"Fourier signal: {T} samples, range [{min(y):.3f}, {max(y):.3f}], variance {var_y:.4f}")
print(f"  f1 = 0.3 (period ~21 steps), f2 = 0.7 (period ~9), f3 = 1.3 (period ~5)")
print(f"  Non-commensurate: combined period >> {T}")

# --- Function fitting: t -> y(t) ---
with open("bench_fourier_ff.csv", "w") as f:
    for t in range(100, T):  # skip first 100 (transient)
        tn = (t - 500) / 200.0  # normalize t to [-2, 2.5]
        f.write(f"{tn:.6f},{y[t]:.6f}\n")
print(f"Wrote bench_fourier_ff.csv: {T-100} samples, 1 input (t), 1 output")

# --- Time series: y(t-1) -> y(t) ---
with open("bench_fourier_ts1.csv", "w") as f:
    for t in range(1, T):
        f.write(f"{y[t-1]:.6f},{y[t]:.6f}\n")
print(f"Wrote bench_fourier_ts1.csv: {T-1} samples, 1 input (y(t-1)), 1 output")

# --- Embedded: [y(t-1), y(t-5), y(t-10)] -> y(t) ---
with open("bench_fourier_ts3.csv", "w") as f:
    for t in range(10, T):
        f.write(f"{y[t-1]:.6f},{y[t-5]:.6f},{y[t-10]:.6f},{y[t]:.6f}\n")
print(f"Wrote bench_fourier_ts3.csv: {T-10} samples, 3 inputs, 1 output")

print(f"\nBaseline: predict-mean MSE = {var_y:.4f}")
print(f"Good predictor target: MSE < 0.01")

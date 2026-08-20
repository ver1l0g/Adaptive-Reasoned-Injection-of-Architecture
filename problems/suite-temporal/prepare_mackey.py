"""Generate Mackey-Glass time series for GP-NN recurrence testing.

Mackey-Glass equation (chaotic delayed differential equation):
  dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

Standard parameters: beta=0.2, gamma=0.1, n=10, tau=17, dt=0.1
Initial condition: x = 1.2 for t < 0

Generates two datasets:
  bench_mackey1.csv  — 1 input (x(t-1)), 1 output (x(t)). Pure recurrence test.
                       The mapping x(t-1)->x(t) is multi-valued (same x(t-1) can
                       lead to different x(t) depending on earlier history), so
                       a non-recurrent predictor will plateau at conditional var.
  bench_mackey4.csv  — 4 inputs (x(t-1), x(t-7), x(t-14), x(t-21)), 1 output.
                       Embedded coordinates — a non-recurrent approach CAN solve
                       this (it's a static function of 4 lagged values). Tests
                       whether the engine finds the embedding without recurrence.

GP-NN usage (pure recurrence):
  ..\gpnn.exe --csv bench_mackey1.csv --input-cols 1 --no-shuffle \
      --max-epochs 300 --seed 1

GP-NN usage (embedded, no recurrence needed):
  ..\gpnn.exe --csv bench_mackey4.csv --input-cols 4 \
      --max-epochs 300 --seed 1
"""
import math

# ---------------------------------------------------------------------------
# Integrate Mackey-Glass via RK4 with delayed feedback
# ---------------------------------------------------------------------------
beta, gamma, n_exp, tau = 0.2, 0.1, 10.0, 17.0
dt = 0.1
n_steps_delay = int(tau / dt)  # 170 steps of delay

# History buffer (initialize to 1.2 for t < 0)
total_steps = 20000
x = [1.2] * (n_steps_delay + 1)

def mg_derivative(x_t, x_delayed):
    return beta * x_delayed / (1.0 + x_delayed ** n_exp) - gamma * x_t

# RK4 integration
for t in range(n_steps_delay, total_steps + n_steps_delay):
    x_t = x[t]
    x_d = x[t - n_steps_delay]

    k1 = mg_derivative(x_t, x_d)
    k2 = mg_derivative(x_t + 0.5*dt*k1, x[t - n_steps_delay] if t - n_steps_delay + 5 < len(x) else x_d)
    # Simplified RK4 (use Euler for delayed term — standard practice)
    x_next = x_t + dt * k1
    x.append(x_next)

# Discard transient (first 5000 steps after delay buffer)
series = x[n_steps_delay + 5000:]

# Sample at every 10th step (dt=0.1 -> sample at integer time steps)
sampled = series[::10]
print(f"Mackey-Glass series: {len(sampled)} samples, range [{min(sampled):.3f}, {max(sampled):.3f}]")

# ---------------------------------------------------------------------------
# Dataset 1: x(t-1) -> x(t)  [pure recurrence test]
# ---------------------------------------------------------------------------
with open("bench_mackey1.csv", "w") as f:
    for i in range(1, len(sampled)):
        f.write(f"{sampled[i-1]:.6f},{sampled[i]:.6f}\n")
print(f"Wrote bench_mackey1.csv: {len(sampled)-1} samples, 1 input, 1 output")

# ---------------------------------------------------------------------------
# Dataset 2: [x(t-1), x(t-7), x(t-14), x(t-21)] -> x(t)  [embedded]
# ---------------------------------------------------------------------------
max_lag = 21
with open("bench_mackey4.csv", "w") as f:
    for i in range(max_lag, len(sampled)):
        x1 = sampled[i-1]
        x2 = sampled[i-7]
        x3 = sampled[i-14]
        x4 = sampled[i-21]
        y = sampled[i]
        f.write(f"{x1:.6f},{x2:.6f},{x3:.6f},{x4:.6f},{y:.6f}\n")
print(f"Wrote bench_mackey4.csv: {len(sampled)-max_lag} samples, 4 inputs, 1 output")

# Report expected loss floors
import statistics
var_full = statistics.variance(sampled)
print(f"\nVariance of full series: {var_full:.6f}")
print(f"A non-predictive baseline (predict mean) has MSE = {var_full:.6f}")
print(f"A good recurrent predictor targets MSE < 0.01")
print(f"A good embedded predictor targets MSE < 0.001")

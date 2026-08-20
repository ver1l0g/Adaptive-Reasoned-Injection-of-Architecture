#!/usr/bin/env python3
"""Temporal memory benchmarks: NARMA-10 and Lorenz (univariate).

Memory stress tests for recurrent capability (BPTT is truncated k=1):
  narma10.csv    input u[t]        -> target y[t]      (needs ~10-step memory)
  narma10_lag.csv inputs u[t..t-9] -> target y[t]      (u-memory handed to the model;
                                                        y-memory still implicit)
  lorenz.csv     input x[t]        -> target x[t+1]    (Takens-style embedding via
                                                        recurrence)
  lorenz3.csv    inputs x,y,z[t]   -> targets x,y,z[t+1] (well-posed control;
                                                        should be easy)

All rows in temporal order — run with --no-shuffle (sequence mode).
"""
import csv, math, os, random

def narma10(n=1400, warmup=200, seed=42):
    rng = random.Random(seed)
    u = [rng.uniform(0.0, 0.5) for _ in range(n + warmup + 11)]
    y = [0.0] * (n + warmup + 11)
    for t in range(10, n + warmup + 10):
        s = sum(y[t-i] for i in range(10))       # y[t-1..t-9] (10 lags)
        y[t+1] = 0.3*y[t] + 0.05*y[t]*s + 1.5*u[t-9]*u[t] + 0.1
    rows_u, rows_lag = [], []
    for t in range(warmup + 11, warmup + 11 + n):
        rows_u.append([f"{u[t]:.6f}", f"{y[t]:.6f}"])
        lag_inputs = [u[t-k] for k in range(10)]
        rows_lag.append([f"{v:.6f}" for v in lag_inputs] + [f"{y[t]:.6f}"])
    return rows_u, rows_lag

def lorenz(n=1400, warmup=500, dt=0.01, every=10, seedunused=None):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    x, y, z = 1.0, 1.0, 1.0
    xs = []
    total = (n*every) + warmup + 10
    for i in range(total):
        # RK4
        def f(x, y, z):
            return (sigma*(y-x), x*(rho-z)-y, x*y - beta*z)
        k1 = f(x, y, z)
        k2 = f(x+k1[0]*dt/2, y+k1[1]*dt/2, z+k1[2]*dt/2)
        k3 = f(x+k2[0]*dt/2, y+k2[1]*dt/2, z+k2[2]*dt/2)
        k4 = f(x+k3[0]*dt, y+k3[1]*dt, z+k3[2]*dt)
        x += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        y += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        z += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        xs.append((x, y, z))
    # rescale to avoid huge magnitudes: divide by 20 (engine z-scores anyway if std>=2)
    rows1, rows3 = [], []
    for i in range(warmup, warmup + n*every, every):
        cur = xs[i]; nxt = xs[i+every]
        rows1.append([f"{cur[0]/20:.6f}", f"{nxt[0]/20:.6f}"])
        rows3.append([f"{cur[0]/20:.6f}", f"{cur[1]/20:.6f}", f"{cur[2]/20:.6f}",
                      f"{nxt[0]/20:.6f}", f"{nxt[1]/20:.6f}", f"{nxt[2]/20:.6f}"])
    return rows1, rows3

def write(fname, header, rows):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  Wrote {fname}: {len(rows)} rows")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Temporal benchmarks:")
    u, lag = narma10()
    write("bench_narma10.csv", ["u", "y"], u)
    write("bench_narma10_lag.csv", [f"u{i}" for i in range(10)] + ["y"], lag)
    l1, l3 = lorenz()
    write("bench_lorenz.csv", ["x", "x_next"], l1)
    write("bench_lorenz3.csv", ["x", "y", "z", "x_next", "y_next", "z_next"], l3)
    print("Done — run with --no-shuffle (sequence mode)")

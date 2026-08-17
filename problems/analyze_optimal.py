import numpy as np
from scipy.optimize import differential_evolution
import csv

rows = list(csv.reader(open(r'C:\Users\banny\Documents\algorithm project\bench1_conditional_linear.csv')))
xs = np.array([float(r[0]) for r in rows])
ys = np.array([float(r[1]) for r in rows])

def tanh_mse(params):
    a, b, c, d = params
    pred = a * np.tanh(b * xs + c) + d
    return np.mean((pred - ys)**2)

bounds = [(0, 30), (0.01, 10), (-5, 10), (-20, 20)]
result = differential_evolution(tanh_mse, bounds, maxiter=1000, seed=42, popsize=30)
a, b, c, d = result.x

print(f'Global optimum single-tanh fit:')
print(f'  a(scale)={a:.6f}, b(weight)={b:.6f}, c(bias)={c:.6f}, d(output_bias)={d:.6f}')
print(f'  MSE per sample (all 201): {result.fun:.6f}')
print(f'  Total loss over 201 samples: {result.fun * 201:.4f}')
print(f'  Train loss (161 samples): {result.fun * 161:.4f}')
print(f'')
print(f'GP-NN best eval loss (averaged per sample): ~18.1')
print(f'Gap vs optimal: {18.1 - result.fun:.4f}')

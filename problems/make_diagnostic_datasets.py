"""Generate diagnostic benchmark datasets for GP-NN.

Each task isolates ONE capability gap. Failure modes are diagnostic:
  d1  step           -> discontinuity / IFELSE gating
  d2  mod3           -> multi-region periodic boundaries (scales beyond 1-2 knots)
  d3  two_sines      -> expressivity / multi-frequency (needs depth)
  d4  irrelevant12   -> feature selection / blackboard search (Phase 4)
  d5  extrap_cliff   -> generalization / extrapolation (train narrow, sweep wide)
  d6  seq_parity     -> recurrence / BPTT (split shuffles -> floor ~0.25 expected)
  d7  multiout       -> multi-output (OUTPUT fan-out, multi-loss)
  d8  compose_absprod-> composition + Phase-6 compression (needs |.| AND product)
  d9  noise          -> overfitting / early stopping (known-solvable + label noise)

All targets kept within [-1, 1] where possible so the NEURON tanh output bound
does not confound the structural diagnostic.
"""
import math
import random

random.seed(42)

# ----------------------------------------------------------------------------
# d1 - Pure step / Heaviside (discontinuity)
#   y = -1 for x < 0,  y = +1 for x >= 0,   x in [-3, 3]
# ----------------------------------------------------------------------------
with open("bench_d1_step.csv", "w") as f:
    for _ in range(200):
        x = random.uniform(-3, 3)
        y = -1.0 if x < 0 else 1.0
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_d1_step.csv")

# ----------------------------------------------------------------------------
# d2 - Modular arithmetic, multi-region periodic (x mod 3) - 1
#   4 cycles over x in [0, 12); targets in {-1, 0, 1}
# ----------------------------------------------------------------------------
with open("bench_d2_mod3.csv", "w") as f:
    for _ in range(240):
        x = random.uniform(0, 12)
        y = float(int(x) % 3) - 1.0
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_d2_mod3.csv")

# ----------------------------------------------------------------------------
# d3 - Sum of two sines (multi-frequency -> needs feature hierarchy / depth)
#   y = sin(3x) + sin(11x), normalized to [-1, 1]; x in [-pi, pi]
# ----------------------------------------------------------------------------
with open("bench_d3_two_sines.csv", "w") as f:
    for _ in range(240):
        x = random.uniform(-math.pi, math.pi)
        y = (math.sin(3 * x) + math.sin(11 * x)) / 2.0  # range [-1, 1]
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_d3_two_sines.csv")

# ----------------------------------------------------------------------------
# d4 - Irrelevant features (feature selection / Phase-4 blackboard search)
#   12 inputs, only x0*x1 matter; x_i in [-1, 1] -> y in [-1, 1]
#   A capable system must identify the 2 relevant columns out of 12.
# ----------------------------------------------------------------------------
with open("bench_d4_irrelevant12.csv", "w") as f:
    for _ in range(300):
        xs = [random.uniform(-1, 1) for _ in range(12)]
        y = xs[0] * xs[1]  # only first two matter
        f.write(",".join(f"{v:.4f}" for v in xs) + f",{y:.4f}\n")
print("Wrote bench_d4_irrelevant12.csv")

# ----------------------------------------------------------------------------
# d5 - Extrapolation cliff (train narrow, sweep wide)
#   y = x^2, train x in [-1, 1] (y in [0, 1]).
#   Run with:  --sweep 1.0 3.0 0.25   to probe outside the training range.
# ----------------------------------------------------------------------------
with open("bench_d5_extrap_cliff.csv", "w") as f:
    for _ in range(200):
        x = random.uniform(-1, 1)
        y = x * x
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_d5_extrap_cliff.csv")

# ----------------------------------------------------------------------------
# d6 - Sequence parity (recurrence / BPTT probe)
#   Bit stream; y_t = running XOR parity of all bits up to t.
#   NOTE: engine's Dataset::split() shuffles rows, destroying temporal order,
#   so the single-bit -> parity mapping becomes ~random. Expected MSE floor
#   ~0.25 (variance of a uniform {0,1} target) if recurrence is unusable.
# ----------------------------------------------------------------------------
random.seed(42)
with open("bench_d6_seq_parity.csv", "w") as f:
    parity = 0
    for _ in range(400):
        bit = random.choice([0, 1])
        parity ^= bit
        f.write(f"{bit:.4f},{float(parity):.4f}\n")
print("Wrote bench_d6_seq_parity.csv")

# ----------------------------------------------------------------------------
# d7 - Multi-output (OUTPUT fan-out, multi-loss)
#   1 input x in [-1, 1]; targets = [x, x^2, sin(x)]  (3 outputs)
#   Run with --output-cols 3.
# ----------------------------------------------------------------------------
random.seed(42)
with open("bench_d7_multiout.csv", "w") as f:
    for _ in range(200):
        x = random.uniform(-1, 1)
        y0 = x
        y1 = x * x
        y2 = math.sin(x)
        f.write(f"{x:.4f},{y0:.4f},{y1:.4f},{y2:.4f}\n")
print("Wrote bench_d7_multiout.csv")

# ----------------------------------------------------------------------------
# d8 - Factorable composition (|x0 * x1|) -> composition + Phase-6 compression
#   Needs product (MULTIPLY) AND abs (IFELSE) discovered as separate sub-circuits.
#   x_i in [-1.5, 1.5] -> |product| in [0, 2.25]; normalized to [0, 1].
# ----------------------------------------------------------------------------
random.seed(42)
with open("bench_d8_compose_absprod.csv", "w") as f:
    for _ in range(240):
        x0 = random.uniform(-1.5, 1.5)
        x1 = random.uniform(-1.5, 1.5)
        y = abs(x0 * x1) / 2.25  # in [0, 1]
        f.write(f"{x0:.4f},{x1:.4f},{y:.4f}\n")
print("Wrote bench_d8_compose_absprod.csv")

# ----------------------------------------------------------------------------
# d9 - Label noise (overfitting / early-stopping probe)
#   Same surface as t23 (x1^2 + x2^2), x_i in [-1, 1] -> y in [0, 2].
#   Add 10% Gaussian noise on y. A system without regularization / early
#   stopping will drive train loss below the noise floor while val loss stalls.
# ----------------------------------------------------------------------------
random.seed(42)
with open("bench_d9_noise.csv", "w") as f:
    for _ in range(200):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        y = x1 * x1 + x2 * x2              # in [0, 2]
        y += random.gauss(0.0, 0.1 * 2.0)  # 10% of full target range (2.0)
        f.write(f"{x1:.4f},{x2:.4f},{y:.4f}\n")
print("Wrote bench_d9_noise.csv")

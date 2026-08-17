"""Generate Tier 2.1-2.4 benchmark datasets for GP-NN."""
import random
import math

random.seed(42)

# ----------------------------------------------------------------------------
# Tier 2.1 — Three-region step function (stacked IFELSE)
#   y = -1   for x < -2
#   y =  0   for -2 <= x <= 2
#   y = +1   for x > 2
# ----------------------------------------------------------------------------
with open("bench_t21_three_region.csv", "w") as f:
    for _ in range(180):
        x = random.uniform(-5, 5)
        if x < -2:
            y = -1.0
        elif x > 2:
            y = 1.0
        else:
            y = 0.0
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_t21_three_region.csv")

# ----------------------------------------------------------------------------
# Tier 2.2 — Windowed sine (IFELSE on raw input + NEURON for sine)
#   y = sin(x)   for x in [0, 2*pi]
#   y = 0        otherwise
# ----------------------------------------------------------------------------
with open("bench_t22_windowed_sine.csv", "w") as f:
    for _ in range(200):
        x = random.uniform(-2, 8)  # spans both zero and sine regions
        if 0 <= x <= 2 * math.pi:
            y = math.sin(x)
        else:
            y = 0.0
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_t22_windowed_sine.csv")

# ----------------------------------------------------------------------------
# Tier 2.3 — Quadratic surface (two repeated products -> ADD)
#   y = x1^2 + x2^2,    x1, x2 in [-2, 2]
# ----------------------------------------------------------------------------
with open("bench_t23_quadratic.csv", "w") as f:
    for _ in range(200):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        y = x1 * x1 + x2 * x2
        f.write(f"{x1:.4f},{x2:.4f},{y:.4f}\n")
print("Wrote bench_t23_quadratic.csv")

# ----------------------------------------------------------------------------
# Tier 2.4 — Quadrant XOR with continuous targets
#   y = x1 * x2   if signs differ (one positive, one negative)
#   y = 0         if signs are the same
# Tests BOOLEAN_COMPOSE (XOR condition) + MULTIPLY_INJECTION (product value)
# ----------------------------------------------------------------------------
with open("bench_t24_quadrant_xor.csv", "w") as f:
    for _ in range(200):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        sign_xor = (x1 > 0) != (x2 > 0)
        y = (x1 * x2) if sign_xor else 0.0
        f.write(f"{x1:.4f},{x2:.4f},{y:.4f}\n")
print("Wrote bench_t24_quadrant_xor.csv")

# ============================================================================
# Tier 3 — Hypothesis stacking & composition
# ============================================================================
# Each problem isolates ONE capability the current hypothesis system hasn't
# been directly tested on. Failure modes are diagnostic: if T3.1 fails, the
# issue is MULTIPLY chaining; if T3.2 fails, it's hypothesis composition
# order; etc.

# ----------------------------------------------------------------------------
# Tier 3.1 — Three-way product (MULTIPLY stacking depth)
#   y = x1 * x2 * x3,    x_i in [-2, 2]
# Needs 2 stacked MULTIPLY nodes: MULTIPLY(x1,x2) then MULTIPLY(.,x3).
# Tests blackboard propagation through intermediate product nodes — after the
# first MULTIPLY commits, does the system recognize its output as a signal
# that should be combined with x3?
# ----------------------------------------------------------------------------
with open("bench_t31_three_way_product.csv", "w") as f:
    for _ in range(200):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        x3 = random.uniform(-2, 2)
        y = x1 * x2 * x3
        f.write(f"{x1:.4f},{x2:.4f},{x3:.4f},{y:.4f}\n")
print("Wrote bench_t31_three_way_product.csv")

# ----------------------------------------------------------------------------
# Tier 3.2 — Sine of product (MULTIPLY → NEURON composition order)
#   y = sin(x1 * x2),    x1, x2 in [-1.5, 1.5]  (product range ~[-2.25, 2.25])
# Needs MULTIPLY(x1,x2) FIRST, then NEURON(tanh) to approximate sin on the
# product. Tests composition ORDER: if NEURON is applied to x1, x2 separately
# first (the easier-looking local fix), the product structure is lost and
# cannot be recovered. The engine must commit MULTIPLY before NEURON.
# ----------------------------------------------------------------------------
with open("bench_t32_sine_of_product.csv", "w") as f:
    for _ in range(200):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        y = math.sin(x1 * x2)
        f.write(f"{x1:.4f},{x2:.4f},{y:.4f}\n")
print("Wrote bench_t32_sine_of_product.csv")

# ----------------------------------------------------------------------------
# Tier 3.3 — Absolute value (IFELSE on raw input as condition)
#   y = |x|,    x in [-3, 3]
# Single IFELSE with condition = INPUT (sign of x), branches = -x and +x.
# Deceptively simple but tests threshold detection on a raw input dimension
# rather than on an intermediate node — the system must select x itself as
# the IFELSE condition_source_node (not a derived feature).
# ----------------------------------------------------------------------------
with open("bench_t33_absolute_value.csv", "w") as f:
    for _ in range(200):
        x = random.uniform(-3, 3)
        y = abs(x)
        f.write(f"{x:.4f},{y:.4f}\n")
print("Wrote bench_t33_absolute_value.csv")

# ----------------------------------------------------------------------------
# Tier 3.4 — 3-input XOR parity (BOOLEAN_COMPOSE stacking)
#   y = x1 XOR x2 XOR x3,    x_i in {0, 1}  (sampled as continuous near 0/1)
# Needs 2 stacked BOOLEAN_COMPOSE nodes (or 1 BOOLEAN + 1 NEURON cleanup).
# Parity is the canonical hard boolean problem; 3-input XOR cannot be solved
# by a single 2-input boolean operation — it requires composition.
# Inputs are sampled as continuous values clustered near 0 and 1 (with small
# noise) so the GREATER thresholding in BOOLEAN_COMPOSE has to actually fire.
# ----------------------------------------------------------------------------
with open("bench_t34_xor3_parity.csv", "w") as f:
    for _ in range(200):
        # Continuous values clustered at 0.1 and 0.9 (small jitter so the
        # problem isn't trivially exact integers — tests thresholding)
        bits = [random.choice([0.1, 0.9]) + random.uniform(-0.05, 0.05) for _ in range(3)]
        # 3-input XOR parity: odd number of "1"s (>= 0.5) -> 1, else 0
        n_high = sum(1 for b in bits if b >= 0.5)
        y = float(n_high % 2 == 1)
        f.write(f"{bits[0]:.4f},{bits[1]:.4f},{bits[2]:.4f},{y:.4f}\n")
print("Wrote bench_t34_xor3_parity.csv")
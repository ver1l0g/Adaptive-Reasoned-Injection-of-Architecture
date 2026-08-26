# Ablation Table (M6.3)

Each mechanism's contribution, measured as the before/after delta on the
task whose failure bought it. "Before" = the system without the
mechanism (either historically or by disabling it); "After" = with it.
All numbers from verified runs in this project's history.

## Structural hypotheses (added capability)

| Mechanism | Task | Before | After | Source |
|-----------|------|--------|-------|--------|
| RECURRENT_XOR + delay-buffer fix | d6 running parity (MSE) | 0.250 | 8.8e-8 | session 1 |
| SIN freq-init (zero-crossing estimate) | d2 mod3 (MSE, 600ep) | 0.636 | 0.006 | session 1 |
| MULTIPLY dedup (pair switching) | t23 quadratic (R²) | 1.30 | 0.0018 | session 1 |
| MULTI_LAYER_STACK (K-wide hidden) | two-spirals (R²) | 0.665 | 0.0077 | session 2 |
| MULTI_LAYER_STACK | checkerboard (R²) | 0.616 | 0.052 | session 2 |
| PARITY_TREE (atomic XOR fold) | xor5d (BCE) | 0.685 | 1e-5 | session 2 |
| DIVIDE_INJECTION (denominator-gated) | I.15.10 (u+v)/(1+uv) (R²) | 0.986 | 0.9998 | session 3 |
| COMPOUND_SIN_PRODUCT (freq-init + zero-gain) | Korns F4 x·sin(x·y) (R²) | 0.742 | 0.979 | session 3 |
| DELAY_LINE (delayed forward edges) | narma30 (R²) | 0.000 | 0.005→0.29 w/ gates | session 4-5 |

## Training & selection (fixed pathologies)

| Mechanism | Task | Before | After | Source |
|-----------|------|--------|-------|--------|
| Val-based model selection | CIFAR gray 5k val BCE | 5.1 (train 1.9) | 3.4 | session 3 |
| SOFTMAX_CE loss (joint output competition) | digits accuracy | 98.5% (BCE) | 99.5% (SCE) | session 4 |
| SOFTMAX_CE + window scaling | charLM w1→w32 (bpc) | flat (BCE era) | 4.535→4.231 | session 4 |
| EMBED_TRUNK (shared dense representation) | charLM w8 (bpc) | 4.295 | 4.070 | session 5 |

## Diagnosis quality (better bottleneck identification)

| Mechanism | Task | Before | After | Source |
|-----------|------|--------|-------|--------|
| Precision-weighted attribution (M7.1) | hetero3 out2/sin (R²) | 0.813 | 0.973 | session 5 |
| Set-guided splits (CART thresholds) | stripes20 boundary | median (wrong) | 0.898 (real) | session 5 |
| Failure library (cross-task negative prior) | stripes20 rerun | re-tries failed families | penalized -0.16 | session 5 |
| Pattern-aware library boost (M7.5b) | library matches | generic +0.2 | family-targeted x2 | session 5 |

## Performance (wall-clock, same results)

| Mechanism | Benchmark | Before | After | Speedup |
|-----------|-----------|--------|-------|---------|
| compute_targets sample cap (200) | I.47.23 | 1249s | 262s | 4.8x |
| train() threading guards | I.47.23 | 241s | 14.5s | 16.6x |
| Shadow-validation subsample (8k) | charLM w8 cycle | ~16 min/cycle | ~4 min/cycle | ~4x |
| val-eval gating (graph_moved) | I.47.23 | (stacked) | 241→ | +8% |

## Bug fixes with measured impact

| Fix | Symptom removed | Impact |
|-----|----------------|--------|
| Enum/hyp_names desync | every hypothesis gate compared wrong values | DELAY_LINE et al. silently misrouted |
| LINEAR weights lost on deserialize | checkpoints loaded at base rate | params 30→10,270 |
| OUTPUT reads only port 0 | committed IFELSE chains invisible | chains survive |
| Bare ADD over IFELSE ports | splits were functional no-ops | gates now train |
| MinGW thread_local heap corruption | silent SCE crashes | SCE usable |

## Honest open gaps (named, not hidden)

| Gap | Task | Current | Named fix |
|-----|------|---------|-----------|
| Long memory | narma30 | 0.29 R² | M4.2 tap composition / deep BPTT |
| Per-branch training | stripes20 | structure solved, values flat | M7.2 settling or per-branch SGD |
| Vision prior | CIFAR | 28% (linear ceiling) | M3.1 shared-weight pooling |
| Deep language mixing | charLM w32 | 4.23 bpc vs bigram 3.54 | M2.3 attention (criteria in 2.4) |
| I.32.8 exact form | q²a²/c³ | 0.967 | M5.2 hypothesis composition |

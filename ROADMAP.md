# ARIA Roadmap

*Adaptive Reasoned Injection of Architecture* — development roadmap from
session evidence (2026-08). Every item traces to a measured failure or a
named gap; nothing here is speculative. Items are grouped by milestone,
ordered within each. Effort: S = hours, M = days, L = weeks.

---

## Current State (the card this roadmap starts from)

| Suite | Status |
|-------|--------|
| suite-standard (17) | 16/17 solved (d9 = noise floor) |
| suite-hard (6) | all solved (spirals, checkerboard, XOR-5D) |
| suite-temporal (4+2) | all solved |
| suite-feynman (25) | 23/25 ≥ 0.99 (I.32.8 0.97, I.29.16 seed-varies) |
| suite-korns (9) | 7/9 ≥ 0.93 (F8 0.84 needs sin∩x³∩4-product) |
| suite-limits (7) | 5/7 pass (narma30 0.29, stripes20 0.24) |
| suite-realworld (5) | 97–100% |
| suite-vision | digits 99.5%; CIFAR at linear ceiling (~28%) |
| suite-language | w1-SCE 4.535 bits/char (unigram 4.78 beaten; bigram 3.54 open) |

Engine: 22 hypothesis types, softmax-CE loss, val-based selection,
checkpointing, shadow watchdog, library with degenerate/self-echo guards.
Performance baseline: I.47.23 1249s → 14.5s cumulative this session (86x).

---

## Milestone 1 — Close the Adaptive Loop (the "A" in ARIA) 
*Theme: ARIA adapts BETWEEN tasks, not just within them. Both builds run
on infrastructure that already works (library, fingerprints, checkpoints).*

### 1.1 Failure library [DONE 2026-08-21]
Persist rejected candidates: (fingerprint, hypothesis family, Δval, task)
triples alongside the success library. The matcher then *down-weights*
proven-wrong families for similar residuals — ARIA stops repeating
mistakes across tasks.
- Data already exists in shadow-validation results; add write path +
  a penalty term in candidate scoring.
- Test: a task family that historically rejects SIN should stop trying
  SIN first (measure: wasted shadow cycles before first commit).

### 1.2 Architecture recall [DONE v1 2026-08-21: checkpoint recall at first plateau, rank -1 shadow]
On a strong library match, don't just boost a candidate — load the
matched entry's *graph* as the initial architecture (checkpoint loading
exists; library entries need graph-JSON fields).
- "Adapting by recall": new task in a known family starts at the old
  solution, converges in a fraction of the budget.
- Test: rerun suite-feynman with warm-start; wall-time per equation
  should drop ≥ 50% on repeats.

### 1.3 Adaptive shadow budget [S]
Track per-family commit rate within a run; allocate shadow-epoch budget
proportional to recent success (10-line bandit / exponential weights).
- Test: hetero3-class runs should stop burning budget on TANH_SERIES
  after it commits 15 times for < 1% gains (already observed pattern).

### 1.4 Self-tuning patience & thresholds [M]
Plateau patience, commit margins, LR schedule adapt to observed plateau
shapes (e.g. patience scales with recent improvement distribution).
- The "adaptive" claim becomes testable: ARIA-vs-ARIA ablation with
  fixed vs tuned constants across the full battery.

---

## Milestone 2 — Break the Language Ceiling
*Theme: dense context mixing. SCE gave correct gradients; the
architecture can't use them. Blocking: ALL sequence understanding.*

### 2.1 EMBED node + trunk hypothesis [M]
New node: K one-hot char slots → dense shared vector (~8 dims/char).
Emission: LM-signature tasks (many one-hot inputs + many outputs) inject
EMBED(per slot) → NEURON trunk → outputs. Standard bigram-model shape.
- Target: w1 ≤ 3.54 bits/char (bigram floor) — measured, crisp.

### 2.2 Window scaling test [S]
With 2.1: rerun the w1→w32 ladder. Success = **monotone improvement
with context width** (currently flat — the defining language failure).
- w8 target ~3.0 (trigram-ish); w32 approaching ~2.5.

### 2.3 (Deferred) Attention-class mixing [L]
Only if 2.2 stalls: dot-product mixing across positions. Big build,
tensor-shaped nodes — genuinely a different node data model. The
induction/copy task is its smoke test. Decision point, not a commitment.

---

## Milestone 3 — Break the Vision Ceiling
*Theme: weight sharing. CIFAR pooling exists but nothing shares weights
across positions.*

### 3.1 Shared-weight pool node [M]
POOL node whose weights are shared across all patches (currently each
pool has independent weights). Halfway to convolution with existing
machinery.
- Test: pre-pooled CIFAR beats raw pixels on validation (the diagnostic
  that never completed due to the hang — rerun it first [S]).

### 3.2 (Deferred) Full conv hypothesis [L]
Stride-2 stacking of 3.1 if it commits. Decision point after 3.1's
verdict — if pooled ≈ raw even with sharing, the bottleneck is
downstream and this is deprioritized.

---

## Milestone 4 — Long Memory (NARMA-30 class)
*Theme: compose short delays into long history. Multi-tap exists (4
taps) but 30-step memory needs mechanism, not more taps.*

### 4.1 Delay-line hypothesis [M]
INPUT history taps (u[t-1..t-k]) as *features* — the narma10_lag trick
(0.005 MSE) built by the engine instead of by the dataset. Emission on
memory-signature plateaus: inject k delay-tapped copies of inputs.
- Test: narma30 (raw) ≥ 0.5 R². (narma10_lag already proves the
  information suffices.)

### 4.2 Tap-composition [M, after 4.1]
Chain multi-tap recurrences so effective memory compounds (taps of taps
= exponential history). Theoretical note: this is fixed-depth RNN
unrolling — the honest limit before attention-style recurrence.

---

## Milestone 5 — Engine Mechanics (parallel track, any time)

### 5.1 Parallel backward pass [M]
The batched backward loop is serial over rev_order; parallelize across
samples like the forward pass. Matters for large-batch runs (CIFAR).

### 5.2 Hypothesis composition at commit [M]
Allow one commit to instantiate a *chain* of templates (e.g.
DIVIDE_PRODUCT exact forms) when the profile uniquely determines it.
I.32.8's remaining gap (0.97 → 0.99) is exactly this.

### 5.3 Diagnosis on multi-output tasks [S–M]
hetero3's out2 (sin output) keeps missing: per-output blame surfaces
the wrong node sometimes. Weight attribution by output-specific error
decomposition (infrastructure exists — d7 solved this way).

### 5.4 Stripes20 / IFELSE family-switching [S]
The fatigue gate fired but PATCH_POOLING spam replaced it. Finish the
family-switch logic: after N failed IFELSE-family commits, force-rotate
to the next family rather than re-scoring the same pool.

---

## Milestone 6 — Scientific Rigor (paper track, starts anytime)

### 6.1 Baselines [M]
Run PySR + a sklearn MLP on suite-feynman/suite-korns with identical
splits. Head-to-head table = the paper's core comparison. SRBench
numbers exist for Feynman — the bar is known.

### 6.2 Multiseed statistics [S]
5-10 seeds on headline claims (harness exists; 2-3 seeds currently).
Mean ± std everywhere; the paper needs error bars.

### 6.3 Ablation table [S–M]
Turn the commit history into a formal ablation: each mechanism's
before/after is already recorded (d6: 0.25→8e-8 after RECURRENT_XOR,
etc.). Structure it as a table; this is the paper's evidence section.

### 6.4 Writing [M]
Method (7-phase loop, 22 hypotheses), results, honest limits (language
ceiling measured, CIFAR ceiling, named open problems), related work
(~15 papers: NAS surveys, NEAT, DARTS, PySR/SRBench, AI Feynman).

---

## Sequencing recommendation

```
Now ──► M1.1 Failure library      (small, closes the loop, the thesis)
    ──► M1.2 Architecture recall  (makes the library visibly pay off)
    ──► M2.1 EMBED                (the biggest single scientific lever)
    ──► M4.1 Delay-line           (cheap, unlocks narma30)
    ──► M6.1 Baselines in parallel (paper track starts here)
    ──► M3/M5 interleaved by interest
    ──► M1.4 Self-tuning          (once the loop is closed, testable)
```

Principles (unchanged all project):
1. Every mechanism is bought by a measured failure — no speculative builds.
2. Every fix is regression-gated (the suite battery is the contract).
3. Honest ceilings are recorded, not hidden — they're the roadmap.

# ARIA Roadmap

*Adaptive Reasoned Injection of Architecture* — development roadmap from
session evidence (2026-08). Every item traces to a measured failure or a
named gap; nothing here is speculative. Items are grouped by milestone,
ordered within each. Effort: S = hours, M = days, L = weeks.

---

## Current State (the card this roadmap starts from)

| Suite | Status |
|-------|--------|
| suite-standard (17) | 12/17 reproducible on aria10 (d9=NF; d7 regressed@381caad fix pending; d2/d3/t22 never-solved, ex-16/17 claim stale — see M5.5) |
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

### 1.3 Adaptive shadow budget [NEGATIVE RESULT 2026-08-27: reverted — see below]
Track per-family commit rate within a run; allocate shadow-epoch budget
proportional to recent success (10-line bandit / exponential weights).
- Test: hetero3-class runs should stop burning budget on TANH_SERIES
  after it commits 15 times for < 1% gains (already observed pattern).

MEASURED (family-fatigue variant, aria12 A/B vs aria11, 3 configurations:
window 4/8 x permanent/25-epoch-parole): suppressing a family whose last
N commits each delivered <1% val gain REGRESSES hetero3 o2 0.998->0.15/0.96
in every config (its sin-harmonic arc is a legit sequence of micro-gain
SIN commits) while stripes20 gains evaporate at window 8 (0.16 only at
window 4, 0.01 otherwise). Val-gain statistics alone cannot distinguish
investment arcs from waste. v2 candidate signal if revisited: per-output
attribution (investment commits concentrate on ONE output; spray grinds
scatter). Code reverted; A/B logs in problems/bisect/a12*/ and
problems/ab_a12*_out.txt.

### 1.4 Self-tuning patience & thresholds [M]
Plateau patience, commit margins, LR schedule adapt to observed plateau
shapes (e.g. patience scales with recent improvement distribution).
- The "adaptive" claim becomes testable: ARIA-vs-ARIA ablation with
  fixed vs tuned constants across the full battery.

---

## Milestone 2 — Break the Language Ceiling
*Theme: dense context mixing. SCE gave correct gradients; the
architecture can't use them. Blocking: ALL sequence understanding.*

### 2.1 EMBED node + trunk hypothesis [DONE 2026-08-26: w8 4.070 bpc (beats w32-linear 4.231); benefit concentrates at small windows]
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
[PC import M7.1 folds in here: precision-weighting the residual before
attribution is the same fix with better justification.]

### 5.4 Stripes20 / IFELSE family-switching [SUBSTANTIALLY DONE 2026-08-25/26: set-guided splits, piecewise signature, PRESERVE+gates, chain protection, port-0 fix — remaining: per-branch training (M7.2 settling or per-branch SGD)]
The fatigue gate fired but PATCH_POOLING spam replaced it. Finish the
family-switch logic: after N failed IFELSE-family commits, force-rotate
to the next family rather than re-scoring the same pool.

---

## Milestone 6 — Scientific Rigor (paper track, starts anytime)

### 6.1 Baselines [MLP DONE 2026-08-26: ARIA 30/34 wins, 3 ties, 1 loss; PySR pending Julia runtime]
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
(~15 papers: NAS surveys, NEAT, DARTS, PySR/SRBench, AI Feynman, and
predictive-coding lineage — see M7 framing).

---

## Milestone 7 — Predictive-Coding Imports (inference-time mechanics)
*Theme: ARIA already IS predictive coding at the architecture-search
timescale (residual-driven diagnosis = error propagation; subgraph
library = generative priors; gain-init = precision weighting). The
stealable remainder is its INFERENCE-TIME mechanics — small, concrete,
and they slot into existing roadmap items rather than a new one. Do NOT
import: free-energy formalism (zero practical yield at this scale).*

### 7.1 Precision-weighted attribution [DONE 2026-08-25: hetero3 out2 0.813->0.973]
Weight each sample's residual by 1/local-variance (precision) before
blame analysis. High-variance regions dilute bottleneck identification —
this is plausibly why hetero3's sin-output keeps getting misdiagnosed.
~20 lines in compute_error_attribution.
- Test: hetero3 out2 (sin) 0.81 → ≥0.95; no regression on d7/multiout.

### 7.2 Settling inference [M] — folds into stripes20 v4
Execute the graph K times per sample, feeding the residual back through
correction paths (predict → error → correct → re-predict). Lets committed
structures pay rent AT INFERENCE TIME instead of waiting for SGD. This
directly attacks the stripes20 finding: each PRESERVE split's benefit
needs training to materialize; settling lets raw structure contribute
immediately.
- Design: a "settle" node type or a train-only loop in execute() —
  recurrent-with-self already has delay-buffer plumbing to reuse.
- Test: stripes20 with multi-split (v3) + settling ≥ 0.5 R² without
  additional SGD budget.

### 7.3 Error-gated correction [S] — folds into M2.1 EMBED
Nodes that compute `predicted − actual` and gate learning downstream
(explicit error units). The killer app: injected corrections proportional
to residual magnitude. DEEP_INSERTION's zero-init identity start is
already `x + correction(·)` — this generalizes it with per-sample gating.
- Test: EMBED trunk with error-gating vs without (same budget), on the
  w32 charLM ladder point (4.231 bits/char baseline).

### 7.4 Promised-vs-delivered credit [S] — strengthens M1.1
After each commit, measure whether it delivered its shadow-promised Δval.
The gap is itself a prediction error (about the FIX) — rank that family
down proportionally. Turns the failure library from per-reject into a
true closed loop: ARIA predicting its own fixes' effectiveness.
- Test: rerun a battery with promised-vs-delivered tracking; wasted
  commits (delivering <50% promised) should decrease in later runs.

### 7.5 Library full-potential audit fixes [S] (from 2026-08-25 audit)
The library stores rich data the engine never reads:
- (a) Skip empty-fingerprint entries in the matcher — the 568
  sub-expression entries (tanh_stack/neuron_unit) pass arity checks
  with meaningless zero-fingerprints and pollute match selection
- (b) Wire the `pattern` field into boost decisions — a "sin_chain"
  match should boost SIN-family candidates specifically, not +0.2
  generic; distance alone is blind to semantic tags
- (c) Parameter seeding from stored canonical expressions — stored
  sin(c*v+c) blocks carry frequencies new tasks need (the freq-init
  mechanism exists; the library is its missing data source)
- (d) Source-diversity weighting in matching — cross-task-family
  matches are the real transfer signal

### 7.6 Set-semantics full-potential [M] (from 2026-08-25 audit)
The set-guided splits compute region memberships then discard them:
- (a) Region membership → MUX branch routing per region (branches
  currently top-2 global signals, blind to the split)
- (b) Per-region residual sign → PRESERVE gate bias init (directly
  attacks the gate-training-slowness from the stripes arc)
- (c) quadrant_means (computed, displayed, never read) → interaction-
  region emission gate: "means differ strongly across quadrants" is
  the natural 2-input interaction signal
- (d) Interval-valued expression rendering for the paper

---


### 6.5 Final freeze battery [S, before writing]
All results in the paper must come from ONE tagged binary on ONE final
battery (suites: standard, limits, feynman, korns, temporal + the
language ladder). Current numbers span evolving builds. Freeze = git tag
+ full run + the numbers become the paper's tables.

### 6.6 README benchmark card refresh [S]
The repo front page undersells current state (22 hypotheses stated,
~25 actual; pre-EMBED language numbers). Update after 6.5 freeze.

### 5.5 Post-arc regression battery [DONE 2026-08-27: root-caused — see results/M5P5_regression_bisect.md]
Verdict: d7_multiout genuinely regressed at 381caad (MUX_INJECTION
crowds out the MULTIPLY_INJECTION emission slot after a poly-fit hit;
a6/a7 commit MULTIPLY at rank 2, a8+ never emit it). Fix pending.
d2/d3/t22 were NEVER solved by any archived binary (all 6 fail in
clean-dir runs; library state and 400 epochs both don't help) — the
16/17 card was session-library-dependent. Reproducible: 12/16 + d9-NF.

### 1.5 Failure-library hypothesis versioning [S]
Discovered in the stripes arc: legacy type-23 (IFELSE_PRESERVE) failures
from v1/v2 runs penalize the v3 multi-split candidates before
validation (-0.16). Fix: version the family ids in failure records, or
bump penalty decay by record age.

### 2.4 Attention decision criteria [S, gates 2.3]
Concrete trigger for the M2.3 go/no-go: run the induction/copy probe
(with EMBED trunk at w32). If bits/char at w32+EMBED stalls above ~3.8
AND the probe fails, attention is justified; if EMBED keeps scaling,
defer again. Defines the measurement, not just "decision point."
---

## Sequencing recommendation

```
Now ──► M1.1 Failure library      (small, closes the loop, the thesis)
    ──► M1.2 Architecture recall  (makes the library visibly pay off)
    ──► M2.1 EMBED + 7.3 error-gating (the biggest scientific lever,
                                       now with the PC ingredient built in)
    ──► M7.1 Precision attribution (cheap, folds into M5.3, fixes hetero3)
    ──► M7.2 Settling + stripes20 v3/v4 (the multi-split test is compiled)
    ──► M4.1 Delay-line           (cheap, unlocks narma30)
    ──► M6.1 Baselines in parallel (paper track starts here)
    ──► M3/M5 interleaved by interest
    ──► M7.4 Promised-vs-delivered (once M1.1 is battle-tested)
    ──► M1.4 Self-tuning          (once the loop is closed, testable)
```

Principles (unchanged all project):
1. Every mechanism is bought by a measured failure — no speculative builds.
2. Every fix is regression-gated (the suite battery is the contract).
3. Honest ceilings are recorded, not hidden — they're the roadmap.
4. (M7 addition) Prefer importing mechanisms that slot into existing
   roadmap items over ones that need a new track — predictive coding's
   gift is inference-time structure, and ARIA's evolution loop already
   has its training-time counterpart.

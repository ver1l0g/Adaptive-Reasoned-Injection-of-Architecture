# ARIA Roadmap

*Adaptive Reasoned Injection of Architecture* — development roadmap from
session evidence (2026-08). Every item traces to a measured failure or a
named gap; nothing here is speculative. Items are grouped by milestone,
ordered within each. Effort: S = hours, M = days, L = weeks.

---

## Current State (card refreshed 2026-08-28/29; binary = aria15 unless noted)

| Suite | Status |
|-------|--------|
| suite-standard (17) | 13/16 + d9-NF reproducible on aria11+ (d7 fixed by M7.5/M7.6 live); t22 0.9757 (routing fixed, was never-solved); d2 0.79, d3 0.55 open — see M5.7 |
| suite-hard (6) | all solved (spirals, checkerboard, XOR-5D) |
| suite-temporal (4+2) | all solved (aria11: 4/4, identical to aria10) |
| suite-feynman (25) | 23/25 ≥ 0.99 (I.32.8 0.97, I.29.16 seed-varies); aria12 re-run: s2 complete, s3-5 queued |
| suite-korns (9) | 6/9 > 0.99, 8/9 ≥ 0.93 (F8 ~0.86 needs sin∩x³∩4-product; F4 0.983) |
| suite-limits (7) | reproducible on aria12: count8 + highdim15 solo (contended runs time out — see M6.7); hetero3 out3 marginal 0.99; narma30 0.005, stripes20 0.2248 on aria15 (region-leaf chains; was 0.01) |
| suite-realworld (5) | 97–100% |
| suite-vision | digits 99.5%; CIFAR at linear ceiling (~28%) |
| suite-language | w1 4.4296 bpc on aria12 (beats 4.535); EMBED ladder w8-w32 in flight (M2.2 verdict pending) |

Engine: 25 hypothesis types (EMBED_TRUNK, MUX_INJECTION, DELAY_LINE +
22 legacy), softmax-CE loss, val-based selection, checkpointing, shadow
watchdog, versioned failure library, zero-plateau evidence path
(single-output v1), SEH crash handler with symbolization, build.bat.
Session repairs: EMBED_TRUNK use-after-realloc (charLM crashes),
PRESERVE chain order, evidence-boundary routing (OUTPUT wrap +
direction), median-fallback threshold clobber.
Known fragility: marginal tasks (hetero3 out2/out3) flip across builds
and sometimes reruns — build/trajectory-sensitive FP, ASAN-clean (see
M6.8).

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

### 2.2 Window scaling test [VERDICT 2026-08-29 (aria12 ladder, EMBED trunk)]
w1 4.4296 → w8 4.2271 → w16 4.1712 bits/char: **MONOTONE IMPROVEMENT
WITH CONTEXT through w16** — the defining language failure ("flat with
width") is broken by the EMBED trunk up to 16 context chars (0.26 bpc
gained w1→w16; accuracy 14.8→22.0%). w32: this run's window collapsed
(6.96 bpc, 0 commits — heaviest shadows watchdog-killed under 5-way
CPU contention; SOLO RERUN PENDING). Historical w32+EMBED (8/26): 4.231
≈ the w8-linear point — no gain at 32. Verdict: monotone scaling
CONFIRMED through w16, ceiling at w32 confirmed by the historical run.

### 2.3 (Deferred) Attention-class mixing [L]
Only if 2.2 stalls: dot-product mixing across positions. Big build,
tensor-shaped nodes — genuinely a different node data model. The
induction/copy task is its smoke test. Decision point, not a commitment.
[GO PER M2.4 CRITERIA — see 2.4: the induction probe failed at chance
(8.01% vs 6.25%) and w32+EMBED stalls at 4.23 > 3.8. Dense fixed
mixing cannot do in-context retrieval; attention is the justified
build. Scope honestly: new node data model (tensor positions),
dot-product mixing, induction/copy as the smoke test.]

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
[UPDATE 2026-08-28 (aria14 evidence path): stripes20 = 20 exact-constant
runs — the zero-plateau detector's ideal target; edges found exactly
(3/cycle, correct directions) and the OUTPUT-wrap routing commits them.
FIRST structural progress: 0.0099 -> 0.0509 with 7 exact boundaries.
Rapid-fire assembly tested and REVERTED (negative): skipping the
inter-cycle gap AND pre-arming plateau patience both REDUCE boundaries
(4-7 vs 7) — the settling epochs between boundary commits are load-
bearing (each boundary's shadow needs the settled state to pass the
drift guard). Cadence is bounded by validation settling, not by gap
or patience mechanics. Path to 20 regions: multi-boundary commits —
the PRESERVE K-chain covers 4 regions/commit — pending the same
wrap/direction routing audit the masking variant just received.]
[CHAIN ORDER FIXED 2026-08-28: the nested PRESERVE chain requires
DESCENDING thresholds (each level splits its false side lower); plateau
edges arrive ASCENDING — feeding them unsorted empties every nested
band and collapses the chain (the persistent 0.27 pre-train
divergence). With descending order the chains now start AT baseline
pre-train (identity restored, 0.238 vs 0.2395) — but SGD degrades them
(0.2499 post-train, rejected by the any-improvement evidence gate).
Chain STRUCTURE is now correct; the blocker is per-gate training —
exactly M7.2 settling / per-branch SGD territory. Boundary singles
still carry stripes20 (7 commits, 0.0509).]
[SOLVED VIA M7.6(b) GENERALIZATION 2026-08-28 (aria15): closed-form
REGION-LEAF chains — each band's true side feeds a 1-input NEURON leaf
(w=1, b = band residual mean, OUTPUT scale/bias adjusted); leaves sum
through chained ADDs to the OUTPUT. The chain becomes a regression
tree AT ROUTING TIME — no SGD needed to express piecewise structure.
stripes20 0.0509 -> 0.2248 (4 chain commits + 5 boundaries).
Sentinels clean (t22 unchanged, t21/d1 PASS). Remaining gap to 0.95:
more cycles / bigger K per chain (edges capped at 3/cycle by
detection), plus whatever narma-style limits emerge.]

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
  [DONE+GENERALIZED 2026-08-28 in aria15: closed-form region-leaf
  chains — per-band residual MEANS (not just sign) init dedicated
  1-input leaf neurons; stripes20 0.01 -> 0.22. See M5.4.]
- (c) quadrant_means (computed, displayed, never read) → interaction-
  region emission gate: "means differ strongly across quadrants" is
  the natural 2-input interaction signal
- (d) Interval-valued expression rendering for the paper

### 6.7 Freeze battery status tracker (added 2026-08-29; binary decision pending)
ONE tagged binary requirement (M6.5). Batteries ran on aria12 (= aria10
+ M7.5/M7.6 live + UAF fix + M1.5); aria14/15 add the evidence path
(detection+routing+chains), which only affects plateau tasks
(t21/d1/t22/stripes20 — standard suite delta). DECISION NEEDED: freeze
on aria12 (all suites consistent, evidence work = post-freeze v2) vs
re-run everything on aria15 (~1-2 days battery).
- [x] standard (aria11: 13/16; aria15/16 spot-checks: t22 0.9757 t21 0.9937 d1 0.9936)
- [x] korns (aria11: 6/9) — [x] temporal (aria11: 4/4)
- [x] limits (aria12 card + highdim20 solo 0.9953 PASS; highdim15 solo
      0.9056 vs aria10 0.9982 — build/trajectory-sensitive, M6.8)
- [x] feynman (aria12 seeds 2-5: mean-of-means 0.9991, **24/25 solved** —
      I.29.16 now solid 0.9993±0.0001; only I.32.8 open at 0.9829±0.0017;
      aria10 archive in results/aria10_multiseed/)
- [x] language ladder (aria12: w1 4.4296 / w8 4.2271 / w16 4.1712 /
      w32 4.4625-solo — monotone through w16, stall at 32 confirmed)
- [x] M2.2 verdict recorded (monotone through w16) + M2.4 (attention GO)
- [x] FREEZE CARD COMPILED: results/FREEZE_CARD.md
      (harness/compile_freeze_card.py — rerunnable aggregator)
- [ ] remaining: git tag freeze-v2 + the aria12-vs-aria16 note for the
      paper's binary statement; s1 on aria12 optional (n=3-4 currently)

### 6.8 Marginal-task build sensitivity (added 2026-08-29, from session evidence)
hetero3 out2/out3 (0.99-bubble tasks) flip between ~0.15 and ~0.999
across builds (aria11 vs aria12: same code semantics, different binary)
and sometimes across identical reruns (0.1503/0.1503/0.1478 — near-
deterministic with small thread variance). ASAN-clean on the full run.
Suspect: thread-order FP nondeterminism in parallel shadow validation,
amplified by marginal tasks. Actions if it matters for the paper:
(1) pin thread count / sequential shadow validation for freeze runs,
(2) report marginal tasks with multiseed spread, (3) investigate the
parallel-validation reduction order.

### 6.9 Reproducibility infrastructure (added 2026-08-29)
- [x] build.bat (MSVC 2022 BuildTools; session-verified flags
      /utf-8 /Zi /O2 /EHsc /std:c++17)
- [ ] pin the toolchain version in docs (MSVC 14.44.35207, x64;
      earlier 16MB binaries used a different/unknown configuration)
- [ ] freeze-run protocol note: solo machine for timing-sensitive
      suites (highdim), thread pinning for marginal tasks (see 6.8)
- [ ] archive binaries WITH their batteries (aria10 freeze-v1 tag
      exists; aria12/15 tags pending the 6.5 decision)

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

### 5.7 Standard-suite never-solved trio — diagnosed 2026-08-27 (analysis complete, fixes unimplemented)
The three tasks no archived binary ever solved (see M5.5 bisect). Log
forensics on aria11 runs pin three DISTINCT engine gaps:
- **d2_mod3 (0.76)**: needs ~6 stacked SIN harmonics; freq-init picks
  the right first freq (w=36.5 from 7 sign changes) but each subsequent
  harmonic fails the strict commit gate individually — value only
  materializes collectively. Same class as hetero3's sin arc (the M1.3
  lesson: val-gain gates cannot price investment arcs). Fix direction:
  harmonic-grace gate (a SIN commit following a SIN commit gets a
  relaxed gate + longer validation train), or COMPOUND_TANH_SERIES with
  K matched to detected knot count.
- **d3_two_sines (0.55)**: freq-init emits ONE mixed-frequency sine
  (w=28.2 — neither true component) from the residual's blended
  zero-crossing rate; it fails validation and dedup blocks retries.
  Fix direction: multi-peak freq-init (autocorrelation or sign-change
  scale decomposition) emitting COMPOUND_TWO_SINE(f1, f2).
  [TWO-PEAK DECOMPOSITION IMPLEMENTED 2026-08-28 (aria15): median-split
  of inter-crossing gaps → w_lo/w_hi, gated on ratio ∈ [1.8, 8] +
  per-cluster CV < 0.8 + component clamp 120. On d3's RESIDUAL the
  gaps are distorted (ratios 60.7/10.5 = tangent-touch noise; the
  blend's own crossings dominate) so it never fires — trajectory
  bit-identical, provably inert. The clean signal exists in the RAW
  LABELS (gap ratio 2.12, w 43.9/93.3). v2 NAMED: label-space two-peak
  (same lesson as the zero-plateau detector — target structure lives
  in the labels, not the post-fit residual).]
  [v2 LABEL-SPACE IMPLEMENTED 2026-08-29 (aria16): the residual
  analysis stays first; on failure the same gap decomposition runs on
  RAW LABELS (single-output). d3: fires perfectly every cycle —
  w_lo=38.3, w_hi=85.8, ratio 2.24, both components emitted. BUT the
  second component never commits (same 2-commit trajectory as the
  blend — the first SIN lands, the w~86 candidate is emitted every
  cycle and never validated). Detection SOLVED; the last mile is the
  same commit-economics wall (high-freq shadow needs longer training
  or SIN re-emission after a SIN commit is being outranked). Next
  lead: SIN-after-SIN grace (mirrors d2's harmonic-grace direction).]
- **t22_windowed_sine (0.95)**: needs IFELSE_BOUNDARY_SPLIT at the
  window edges (x=0, 2pi) wrapping a sine. [ZERO-PLATEAU DETECTOR
  IMPLEMENTED 2026-08-27 in aria13: detection VERIFIED — flat-run scan
  on raw labels finds both edges exactly (normalized input space, z-
  scored since std>2); both emitted as boundary candidates at top
  score; PRESERVE multi-split consumes them directly. REMAINING
  BLOCKER: the boundary candidates lose the shadow-validation race to
  micro-gain TANH/MULTIPLY commits every cycle — val subsample already
  at 0.0087 while eval R2 stays 0.95 (edge error invisible to the
  strided val). Third observed instance of the investment-arc problem:
  short-SGD shadow validation cannot price commits whose payoff needs
  post-commit training (d2 harmonics, hetero3 sin arc, t22 boundaries).
  Fix direction now clear: shadow validation must include a boundary-
  commit grace budget (train masked shadows longer), or accept-as-
  epsilon commits for candidates with exact structural evidence.]
  d9_noise unaffected (detector correctly silent); t21/d1 pass with it.
  [ATTEMPTED 2026-08-27 (aria14, 4 variants — NEGATIVE): structural_
  evidence flag plumbed Hypothesis→specs→validate_shadow_only with
  (a) 3x shadow budget, (b) any-improvement gate, (c) 0.1%-neutral
  tolerance, (d) 10-rank-step race advantage; PRESERVE multi-split
  routing consumes plateau edges directly (verified: "[PRESERVE-MULTI]
  built K=2 split chain"). STILL zero boundary commits and R2 pinned
  at 0.9518 — the rejection happens at a layer ABOVE the accept gate
  (drift guards? family suppression? specs cap?). Next lead: raise log
  verbosity on evidence candidates and read the exact reject_reason.
  Also noted: IFELSE_BOUNDARY_SPLIT always masks x>thr — for a LEFT
  plateau edge this masks the entire active region (correctly
  rejected); only PRESERVE (both-branches) fits windowed targets.
  Evidence plumbing kept in aria14 (inert without plateaus; t21/d1/d9
  sentinels pass).]
  [ROOT CAUSE FOUND 2026-08-28 (aria14 + EVIDENCE-REJECT logging): the
  boundary candidates' VAL IS ALREADY PERFECT (val=0.000000 on the
  masked graph — the structure is correct!) but the shadow TRAINING
  diverges: shadow_train 0.17-0.28 vs baseline 0.009, rejected by the
  drift guard. SGD on the discontinuous masked graph (IFELSE boundary
  + fresh gates + momentum) breaks the in-window fit during the
  extended budget; LR reduction does not stabilize it. This is a
  TRAINING-STABILITY problem (same family as the stripes-arc gate
  training), NOT validation economics as first hypothesized. Fix
  direction: freeze non-gate weights during boundary-shadow training,
  or per-gate gradient clipping, or train gates only (structure is
  measured — only the gates need learning).]
  [CONFIRMED + PARTIAL FIX 2026-08-28: evidence shadows at 0.1x LR —
  the diagnosis was right. First-ever IFELSE_BOUNDARY_SPLIT commit on
  t22 (val 0.0055, drift guard passed); R2 0.9518 -> 0.9610. Still
  below 0.99: one boundary is not enough (PRESERVE multi-split chains
  still diverge at 0.27 train — momentum suspected), and the interior
  sine precision remains the original limiter. Sentinels clean (t21
  0.9921, d1 0.9977, d9 silent).]
  [TRUE ROOT CAUSE 2026-08-28 (EVIDENCE-PRE instrumentation): the
  split shadows are broken BEFORE any SGD — pre-train loss 0.12-0.28
  vs baseline 0.009. The "identity start" of the boundary/PRESERVE
  routing does not exist: apply_shadow_routing's IFELSE wiring corrupts
  the in-window forward pass at ROUTING time (val looks perfect only
  because the strided val subset is dead-zone dominated, where the
  mask outputs exact 0). LR/momentum changes were treating symptoms.
  This is a ROUTING SEMANTICS bug — same family as the stripes-arc
  port-0 ADD bug.]
  [ROUTING FIXED 2026-08-28 (aria14): hand-execution of the dumped
  shadow graph pinned the exact bug — the failing node was ONE of two
  parallel contributors (starter ∥ parallel → ADD → OUTPUT); masking
  an internal contributor zeroes its share while the OUTPUT scale was
  calibrated for the SUM. Evidence boundaries now (a) wrap the FINAL
  signal (OUTPUT's port-0 source) instead of the failing node, (b)
  carry a DIRECTION per edge (a run's left edge masks above, right
  edge masks below — detect_zero_plateau_edges returns tagged pairs),
  and (c) a masked-out median-fallback else-branch that silently
  clobbered measured thresholds with blackboard medians (thr -1.077
  → 0.0376) is guarded. RESULT: t22 0.9518 -> 0.9757 with FOUR
  boundary commits accumulating (val 0.15 -> 0.003); remaining gap is
  interior-sine fit precision, not structure. Sentinels clean (t21
  0.9921, d1 0.9977, d9 detector silent). PRESERVE multi-split routing
  still wraps the failing node — same audit pending for stripes20.]

### 1.5 Failure-library hypothesis versioning [DONE 2026-08-27 in aria12: FAILURE_FAMILY_VERSION=1 written per record; legacy (v0/unversioned) records discount to 25% penalty; loader back-compat]
Discovered in the stripes arc: legacy type-23 (IFELSE_PRESERVE) failures
from v1/v2 runs penalize the v3 multi-split candidates before
validation (-0.16). Fix: version the family ids in failure records, or
bump penalty decay by record age.

### 2.4 Attention decision criteria [MEASURED 2026-08-29: ATTENTION JUSTIFIED]
Concrete trigger for the M2.3 go/no-go: run the induction/copy probe
(with EMBED trunk at w32). If bits/char at w32+EMBED stalls above ~3.8
AND the probe fails, attention is justified; if EMBED keeps scaling,
defer again. Defines the measurement, not just "decision point."
[RESULT: probe (harness/run_induction.py, V=16/W=32, 8k train,
EMBED trunk): accuracy 8.01% vs 6.25% chance — FAIL (dense mixing
cannot retrieve the previous occurrence of the queried token from
context; eval 3.813 bpc). w32+EMBED historical 4.231 > 3.8 stall
threshold (this run's w32 collapsed on watchdog kills — solo rerun
pending, but the 8/26 number stands as the valid measurement). BOTH
criteria met → M2.3 attention build is GO.]
---

## Sequencing recommendation (refreshed 2026-08-29)

```
Now ──► M6.5 freeze completion: ladder + feynman s3-5 + highdim solo
        rerun; make the aria12-vs-aria15 binary decision; compile the
        freeze card; tag.
    ──► M2.2/M2.4 verdicts from the ladder + induction probe (the
        attention go/no-go — decides M2.3's fate)
    ──► d2/d3 fixes (M5.7: harmonic-grace gate; multi-peak freq-init)
    ──► stripes20 completion (bigger K per chain or higher edge cap;
        0.22 -> 0.95 target)
    ──► M6.6 README refresh (after freeze)
    ──► M6.4 Writing (paper track; freeze numbers are the tables)
    ──► M5.2 commit composition (I.32.8 0.97->0.99) / M4.2 taps
    ──► M3.1 shared-weight pooling (vision track)
    ──► M7.4 promised-vs-delivered (post-freeze; mind the M1.3 lesson)
```

Principles (unchanged all project):
1. Every mechanism is bought by a measured failure — no speculative builds.
2. Every fix is regression-gated (the suite battery is the contract).
3. Honest ceilings are recorded, not hidden — they're the roadmap.
4. (M7 addition) Prefer importing mechanisms that slot into existing
   roadmap items over ones that need a new track — predictive coding's
   gift is inference-time structure, and ARIA's evolution loop already
   has its training-time counterpart.

# ARIA

**Adaptive Reasoned Injection of Architecture** — a hybrid of neural
network training and genetic programming. ARIA maintains a single
computation graph, trains it with Adam/SGD, and when gradient descent
plateaus, it *diagnoses why* — blame analysis, target propagation,
residual profiling — then surgically injects a small, targeted
sub-structure. Every candidate structure is validated on held-out data
in a shadow copy before commit; wrong guesses cost nothing.

> **Note on authorship**: this project is AI-implemented — the engine,
> benchmarks, harnesses, and analysis were developed in collaboration
> between the repository owner and an AI coding agent (opencode /
> GLM), with the human directing strategy, reviewing results, and
> making all engineering decisions. The development history (git log,
> ROADMAP.md) records the process, including AI-caused bugs and their
> forensics.

```
train → plateau → diagnose → hypothesize → shadow-validate → commit/reject → repeat
```

## Build (MSVC 2022)

```
build.bat aria.exe          (wraps vcvars64 + cl; pinned flags in BUILD_REPRO.md)
```

Manual: `cl /nologo /utf-8 /Zi /O2 /EHsc /std:c++17 src\main.cpp
src\evolution.cpp src\graph.cpp src\node.cpp src\logger.cpp
src\serialize.cpp src\subgraph_library.cpp /Fe:aria.exe`

Tests: `cl /utf-8 /O2 /EHsc /std:c++17 problems\tests\main_tests.cpp
src\graph.cpp src\node.cpp (minus main) /Fe:tests.exe`

## Use

```
aria.exe --csv data.csv --input-cols N [--output-cols M]
         [--loss mse|bce|sce] [--max-epochs N]
         [--eval-csv heldout.csv] [--no-shuffle]
         [--config configs/default.json]     # 21 tunables; CLI > JSON > defaults
         [--quiet]                           # decision-critical log lines only
         [--dump-graph] [--load-graph path] [--save-graph dir|none]
         [--seed N] [--verbose]
```

After training ARIA prints the evolved graph as a readable symbolic
expression, e.g. `2.161*(1.302*tanh(0.878*x0 + 1.794*x2 + ...))`.

Run batteries: `problems/harness/run_pool.ps1 -TaskFile
problems/harness/tasks.txt -Slots 5` (slot-capped parallel runner).
Status of everything: `powershell -File problems/harness/status.ps1`.

## How it works

- **Graph**: 30+ node types (arithmetic, tanh/sin neurons, comparators,
  logic gates, IF/IFELSE/MUX, ONEHOT expansion, single-head ATTENTION
  with exact per-port gradients) executed by dirty-flag wavefront
  propagation; cycles become recurrent edges with delay buffers
  (truncated BPTT).
- **Diagnosis**: perturbation blame analysis finds bottleneck nodes;
  target propagation computes what each node should have output;
  precision-weighted (1/local-variance) residuals keep blame honest on
  heteroscedastic tasks.
- **Hypotheses** (26 structural templates): MULTIPLY/DIVIDE feature
  injection (scale-aware LINEAR gain), SIN with frequency init from
  zero-crossing + label-space two-peak decomposition, EMBED_TRUNK with
  one-hot expansion for LM-signature tasks, IFELSE_PRESERVE
  region-leaf chains (closed-form regression-tree init), zero-plateau
  evidence boundaries (label-space edge detection with directional
  masking), PATCH_POOLING, and more.
- **Cross-run memory**: behavioral fingerprint library + versioned
  failure library (negative experience, legacy-discounted); matched
  entries boost the proven family.
- **Arc-pricing (M6.12)**: families whose commits stop MOVING the
  residual fingerprint while delivering nothing are demoted (parole
  after 25 epochs) — investment arcs (sustained movement) keep full
  budget. Calibrated on d2-ladder vs stripes20-spray.
- **Safety**: shadow validation with scaled commit gate, wall-clock
  val budget, degenerate-loop suppression, structural cooldown,
  divergence restore, SEH crash handler with symbolized stack dumps.

## Benchmarks (problems/ — all reproducible via prepare_*.py)

Full tables: `problems/results/FREEZE_CARD.md` (feynman multiseed with
error bars; per-suite cards). Headlines:

| Suite | Result |
|-------|--------|
| Feynman equations (25) | **25/25-capable**: 24/25 on the freeze binary; I.32.8 = 1.000000 on 4/5 seeds with the linear-gain fix |
| 17 synthetic tasks | 13/16 + d9 noise-floor (t22 0.976 via evidence boundaries; d2/d3 open with diagnosed fixes) |
| Korns (9) | 6/9 ≥ 0.99 (F8/F4/F9 open) |
| NARMA-10 / Lorenz | 4/4 solved |
| Language (Shakespeare) | EMBED trunk: w1 4.43 → w16 4.17 bits/char — **monotone with context through w16** (the "flat with width" failure broken); ceiling at w32 confirmed; attention GO per pre-registered criteria |
| MNIST 8x8 / Iris / Wine / Breast-Cancer | 97-100% |
| Tang 五言绝句 (suite-poems) | position-entropy baselines measured; structure-induction test bed |
| Two-spirals, checkerboard, XOR-5D | solved |

Known limits (measured, not hidden — see ROADMAP): induction/copy
retrieval at 8.1% (attention head integrates and commits but a single
1-layer head doesn't solve it — 7-config isolation matrix); CIFAR-10
at the raw-pixel linear ceiling (~28%); narma30/stripes20 open;
marginal tasks flip across builds (trajectory-sensitive FP).

## Layout

```
src/        engine (nodes, graph, evolution, serialization, library)
problems/   suites + harnesses + results (FREEZE_CARD.md) + archives
configs/    default.json — the 21 documented run tunables
binaries/   archived build lineage (aria → aria30)
ROADMAP.md  the full development record: every mechanism's evidence,
            negative results, and open items
```

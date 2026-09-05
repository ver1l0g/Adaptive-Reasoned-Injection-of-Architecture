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

## Build (MinGW-w64)

```
g++ -O2 -std=c++17 -static -static-libgcc -static-libstdc++ \
    src/node.cpp src/graph.cpp src/evolution.cpp src/logger.cpp \
    src/serialize.cpp src/subgraph_library.cpp src/main.cpp -o aria.exe
```

Tests: `g++ -O2 -std=c++17 problems/tests/main_tests.cpp src/*.cpp (minus main) -o tests.exe`

## Use

```
aria.exe --csv data.csv --input-cols N [--output-cols M] [--loss mse|bce]
         [--max-epochs N] [--eval-csv heldout.csv] [--no-shuffle]
         [--dump-graph] [--load-graph path] [--save-graph dir|none]
         [--seed N] [--verbose]
```

After training ARIA prints the evolved graph as a readable symbolic
expression, e.g. `2.161*(1.302*tanh(0.878*x0 + 1.794*x2 + ...))`.

## How it works

- **Graph**: ~30 node types (arithmetic, tanh/sin neurons, comparators,
  logic gates, IF/IFELSE) executed by dirty-flag wavefront propagation;
  cycles become recurrent edges with delay buffers (truncated BPTT).
- **Diagnosis**: perturbation blame analysis finds bottleneck nodes;
  target propagation computes what each node should have output.
- **Hypotheses** (20 structural templates): MULTIPLY/DIVIDE feature
  injection, SIN with frequency init from zero-crossing analysis,
  MULTI_LAYER_STACK (solved two-spirals), PARITY_TREE (solved 5-bit
  parity), RECURRENT_XOR (solved running parity), PATCH_POOLING (coarse
  conv prior), and more.
- **Safety**: shadow validation with scaled commit gate, degenerate-loop
  suppression, structural cooldown, divergence restore, validation-based
  model selection, periodic checkpointing.

## Benchmarks (problems/ — all reproducible via prepare_*.py)

| Suite | Result |
|-------|--------|
| 17 synthetic tasks | 16 solved (d9 at its noise floor) |
| Feynman equations (25) | 24 ≥ 0.98 R² |
| Korns (9) | 7 ≥ 0.93 |
| NARMA-10 / Lorenz | 4/4 solved |
| MNIST 8x8 / Iris / Wine / Breast-Cancer | 97-100% |
| Two-spirals, checkerboard, XOR-5D | solved |

Known limits: language modeling (scalar nodes lack dense context
mixing), CIFAR-10 at the raw-pixel linear ceiling (~28%), quotient-of-
products needs scale-aware init (I.32.8 at 0.96).

## Layout

```
src/      engine (nodes, graph, evolution, serialization, library)
problems/ benchmark prep + harnesses + background chains
```

# problems/ — Directory Map

Every benchmark/experiment has a related folder. Nothing loose at root.

## Suites (one folder per benchmark family — data + prep script)
| folder | contents |
|--------|----------|
| suite-standard/ | d1-d9, t21-t34 (17 synthetic tasks) |
| suite-hard/ | spirals, checkerboard, XOR-5D |
| suite-limits/ | highdim, narma30, hetero3, stripes20, count8, firstpos8 |
| suite-temporal/ | narma10 (+lag), lorenz |
| suite-language/ | Shakespeare w1-w32 |
| suite-poems/ | Tang 五言绝句 + corpus (raw/cprepo) |
| suite-vision/ | digits, CIFAR gray/rgb/pooled (+cifar10 source) |
| suite-feynman/ | 25 physics equations |
| suite-korns/ | F1-F9 |
| suite-realworld/ | iris, wine, breast-cancer, titanic |
| suite-signals/ | chirp, fourier |

## Experiment families
| folder | contents |
|--------|----------|
| induction/ | M2.4 attention probes (a18-a25l run dirs) |
| bisect/ | A/B forensics (a0-a16 binary matrix, zp/cfg probes, i62b/i2916) |
| results/ | all CSV results + FREEZE_CARD.md + reports |

## Run history (scratch)
- scratch/ — loose one-off logs
- scratch/runs/ — completed run batteries (arcprobe, ladder12, feyn26,
  i328ab, vision_out, lang_out, hd16...)

## Infrastructure
- harness/ — run_*.py/bat/ps1 drivers, status.ps1, run_pool.ps1,
  tasks.json registry, compile_freeze_card.py
- tests/ — C++ unit tests
- archives/ — pre-session archives (2026-08 era)

## Root files (working state, intentionally at problems/)
- subgraph_library.txt / failure_library.txt — the cross-run memory
- tasks.json — --task registry
- frz11/frz12_*.txt — freeze battery stdout (FREEZE_CARD reads these)
- README_SUITES.md / README_OVERNIGHT.txt / RUN_QUEUE.bat — docs + legacy queue

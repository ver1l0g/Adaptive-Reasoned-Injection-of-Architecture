# Build & Reproduction Protocol (M6.9)

## Toolchain (pinned 2026-09-05)

- **Compiler**: MSVC 19.44.35207 (x64) — Visual Studio 2022 BuildTools
  - Path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`
- **Flags** (session-verified): `/nologo /utf-8 /Zi /O2 /EHsc /std:c++17`
  - `/utf-8` required (sources contain UTF-8 comments)
  - `/Zi` gives the SEH crash handler symbolization (crash_stack.txt)
- **Build**: `build.bat <name>.exe` from the repo root (wraps vcvars64 + cl)

Note: binaries before aria11 (~16MB) were built with an unknown/
different configuration; aria11+ (~1-3MB) are the reproducible lineage.

## Run configuration (M5.8)

`aria.exe --csv data.csv --input-cols N --config configs/default.json`
- Precedence: CLI flags > JSON > compiled defaults
- All 21 tunables documented in `configs/default.json`
- Runs are reproducible from (exe, data, json) triples — no recompiles

## Freeze-run protocol (lessons M6.7/M6.8)

1. **Parallelism**: 5 slots for light suites; **3-4 for watchdog-heavy
   runs** (V>=500 EMBED trunks, big shadows) — 5-way contention starves
   the 900s shadow watchdog (w32 ladder collapse, highdim no-evals).
   Use `harness/run_pool.ps1 -TaskFile harness/tasks.txt -Slots N`.
2. **Timing-sensitive tasks run SOLO**: highdim15/20 (contention doubles
   wall time past the harness 30-min timeout).
3. **Marginal tasks** (hetero3 out2/out3, ~0.99 bubble): report with
   multiseed spread — they flip across builds/reruns (ASAN-clean;
   suspected thread-order FP nondeterminism). Consider sequential
   shadow validation if error bars matter for the paper.
4. **Crash forensics**: the SEH handler writes `crash_stack.txt` in the
   process CWD (symbolized) — check it before assuming a hang.
5. **Abort-proof runs**: launch via `.bat` wrappers (survive shell
   kills); logs are crash-safe tee'd; results CSVs are incremental.

## Binary lineage (tags)

- `freeze-v1` → aria10 (first freeze attempt)
- `freeze-v2` → aria12 (sha256 in the tag message; FREEZE_CARD.md tables)
- aria13-25: post-freeze development (evidence path, ONEHOT, attention,
  config JSON) — v3 candidates, NOT in the freeze card

# M5.5 Post-arc regression battery — bisect findings (2026-08-27)

Freeze battery (aria10, clean dirs, seed 1, 100 epochs) exposed 5
standard-suite failures vs the README "16/17 solved" card. Bisect across
ALL archived binaries (aria.exe 8/17 -> aria10 8/25), each in an isolated
directory (fresh library), 5 suspect tasks:

| task          | a0  | a6  | a7  | a8  | a9  | a10 | @400ep(a10) |
|---------------|-----|-----|-----|-----|-----|-----|-------------|
| d2_mod3       | .06 | .06 | .06 | .06 | .06 | .21 | .92 FAIL    |
| d3_two_sines  | .36 | .36 | .34 | .36 | .50 | .11 | .11 FAIL    |
| d7_multiout   | 1.0 | 1.0 | 1.0 | .34 | .34 | .00 | -           |
| t21_three_reg | 1.0 | 1.0 | 1.0 | 1.0 | .91 | .99 | -           |
| t22_windowed  | .97 | .97 | .97 | .97 | .97 | .97 | .97 FAIL    |

## Finding 1 — d7 is a genuine binary regression (root-caused)

Window: aria7 (8/19 21:16) -> aria8 (8/20 19:10). Engine commits in the
window: 75c9bd4 (SOFTMAX_CE) and 381caad (MUX node + MUX_INJECTION).

Mechanism (log diff a6 vs a8, d7 = targets [x, x^2, sin(x)], failing
output is out1 = x^2):

- BOTH binaries log `MULTIPLY sources from poly fit: input=1^2` — the
  poly-fit detector fires correctly.
- a6 then emits MULTIPLY_INJECTION (1 mention) and COMMITs it at rank=2:
  val 0.0808 -> 0.0090. Run ends 0.9999.
- a8 emits it ZERO times. Instead MUX_INJECTION candidates fire every
  cycle (5+ emissions, all pre-commit noise), CONTEXT_WIRE takes rank=0
  for a 0.0002 gain, and TANH_SERIES micro-grinds to the epoch cap.
  Run ends 0.34 (out1 unsolved).

Conclusion: 381caad's new MUX family crowds out the MULTIPLY_INJECTION
emission slot after a poly-fit hit — the candidate never enters the
pool. This is the M5.5 "post-arc regression" the roadmap predicted
(though it predates the stripes arc). Fix direction: MUX emission must
not consume the poly-fit MULTIPLY slot (separate slots or
detector-gated priority); verify d7 returns to 0.9999 and MUX tasks
(suite-limits hetero3/stripes20, t21) do not regress.

## Finding 2 — d2/d3/t22 were NEVER solved by any archived binary

All six binaries (including aria.exe, the oldest) fail these three in
clean-directory runs. Dataset generators unchanged since 8/17 (reorg
d9a0636 was a pure rename). Library state exonerated: a10lib (shared
1.4MB library + failure library copied in, confirmed loaded: "library
now has 1001 entries") produces results IDENTICAL to a10clean to 4
decimals. 400 epochs does not rescue (d2 .92, d3 .11, t22 .97).

The "16/17 solved" session claim is therefore not reproducible on any
archived artifact — it presumably depended on the live session's
accumulated library at a golden state (or a config not recorded).
Reproducible current count on aria10: 12/16 pass + d9 noise-floor
(t21 at 0.9902 marginal). With the d7 fix: 13/16.

Honest options: (a) fix d7, re-run, and update the card to the
reproducible number; (b) attack d2/d3/t22 as open problems (d2 needs
multi-knot periodic IFELSE — the stripes-arc machinery may now solve
it; d3 needs multi-frequency SIN depth; t22 needs IFELSE+sine
composition — 0.97 across all builds suggests a systematic near-miss).

## Artifacts

- bisect/<tag>/results.csv + per-task logs (tags: a0clean a6clean a7clean
  a8clean a9clean a10clean a10lib)
- harness/bisect_standard.py (the bisect driver)
- harness/probe_epochs.py + probe_*_e400.txt (epoch sensitivity)
- frz_*_out.txt (freeze battery stdout), results/standard_results.csv

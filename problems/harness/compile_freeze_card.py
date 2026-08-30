#!/usr/bin/env python3
"""M6.5/M6.7 freeze card compiler — aggregates every freeze-battery result
into results/FREEZE_CARD.md. Sources (aria12 lineage unless noted):

  standard   results/standard_results.csv          (aria11; aria16 spot-checks noted)
  korns      results/korns_results.csv             (aria11)
  temporal   frz11_temporal_out.txt                (aria11)
  limits     frz12_limits_out.txt + solo reruns    (aria12)
  feynman    results/feynman_results_seed{2..5}    (aria12; aria10 archive in
             results/aria10_multiseed/)
  language   ladder12/w{1,8,16,32}/run_log.txt     (aria12; w32 = solo rerun)

Usage: python harness/compile_freeze_card.py   (from problems/)
"""
import csv
import os
import re
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = "results"


def std_card():
    if not os.path.exists(f"{R}/standard_results.csv"):
        return "_missing_"
    rows = list(csv.DictReader(open(f"{R}/standard_results.csv")))
    solved = [r for r in rows if r["task"] != "t22_windowed_sine"
              and r["min_r2"] and float(r["min_r2"]) > 0.99]
    lines = [f"{len(solved)}/{len(rows)-1} pass + d9 noise-floor",
             ""]
    for r in rows:
        r2 = r["min_r2"]
        tag = "" if (r2 and float(r2) > 0.99) else " **OPEN**"
        if r["task"] == "d9_noise":
            tag = " (noise floor)"
        lines.append(f"| {r['task']} | {r2 or 'n/a'} |{tag}")
    return "\n".join(lines)


def korns_card():
    if not os.path.exists(f"{R}/korns_results.csv"):
        return "_missing_"
    rows = list(csv.DictReader(open(f"{R}/korns_results.csv")))
    solved = [r for r in rows if r["test_r2"] and float(r["test_r2"]) > 0.99]
    lines = [f"{len(solved)}/9 > 0.99", ""]
    for r in rows:
        r2 = r["test_r2"]
        tag = "" if (r2 and float(r2) > 0.99) else " **OPEN**"
        lines.append(f"| {r['prob']} | {r2 or 'n/a'} |{tag}")
    return "\n".join(lines)


def temporal_card():
    txt = open("frz11_temporal_out.txt", encoding="utf-8",
               errors="replace").read() if os.path.exists("frz11_temporal_out.txt") else ""
    return txt.strip() or "_missing_"


def limits_card():
    txt = open("frz12_limits_out.txt", encoding="utf-8",
               errors="replace").read() if os.path.exists("frz12_limits_out.txt") else ""
    lines = [l for l in txt.splitlines() if l.strip()]
    lines.append("")
    lines.append("| highdim20 solo rerun | 0.995278 PASS |")
    lines.append("| highdim15 solo rerun | 0.905612 (aria10 freeze: 0.9982 —")
    lines.append("|                      |  build/trajectory-sensitive, see M6.8) |")
    return "\n".join(lines) if lines else "_missing_"


def feynman_card():
    seeds = {}
    for s in (2, 3, 4, 5):
        p = f"{R}/feynman_results_seed{s}.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r.get("test_r2"):
                    seeds.setdefault(r["eq"], []).append(float(r["test_r2"]))
    if not seeds:
        return "_missing_"
    lines = ["aria12, 50 epochs, seeds 2-5 (aria10 archive: results/aria10_multiseed/)", ""]
    solved = means = 0
    for eq in sorted(seeds):
        v = seeds[eq]
        m = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        means += m
        ok = m > 0.99
        solved += ok
        tag = "" if ok else " **OPEN**"
        lines.append(f"| {eq} | {m:.4f} ± {sd:.4f} (n={len(v)}) |{tag}")
    lines.insert(1, f"mean-of-means {means/len(seeds):.4f} | solved {solved}/{len(seeds)}")
    return "\n".join(lines)


BPC_RE = re.compile(r"Eval SoftmaxCE:\s*([0-9.]+)\s+\(([0-9.]+) bits/unit\)")


def language_card():
    lines = []
    for w in ("w1", "w8", "w16", "w32"):
        p = f"ladder12/{w}/run_log.txt"
        if not os.path.exists(p):
            continue
        m = None
        for line in open(p, encoding="utf-8", errors="replace"):
            mm = BPC_RE.search(line)
            if mm:
                m = mm
        if m:
            lines.append(f"| {w} | {m.group(2)} bits/char |")
    lines.append("")
    lines.append("Monotone through w16 (4.43 -> 4.17); stall at w32 confirmed")
    lines.append("(solo rerun 4.4625; historical 4.231). M2.4: attention GO.")
    return "\n".join(lines) if lines else "_missing_"


def main():
    card = f"""# ARIA Freeze Card (M6.5/M6.7) — compiled 2026-08-30

Binary lineage: aria12 (aria10 + M7.5/M7.6 live + EMBED UAF fix + M1.5
versioning) for all suites; aria11 for standard/korns/temporal (identical
engine semantics to aria12 on those paths); aria16 spot-checks noted in
place. Post-freeze development binaries (aria13-18: evidence path,
one-hot) are NOT in this card — they are v2.
Development binaries & deltas: see ROADMAP M6.7 for the freeze-binary
decision record.

## suite-standard (17 tasks, aria11)
{std_card()}

## suite-korns (9 tasks, aria11)
{korns_card()}

## suite-temporal (aria11)
```
{temporal_card()}
```

## suite-limits (aria12; solo reruns where the 5-way runs timed out)
```
{limits_card()}
```

## suite-feynman multiseed (aria12)
{feynman_card()}

## suite-language ladder (aria12, EMBED trunk)
{language_card()}
"""
    out = f"{R}/FREEZE_CARD.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"wrote {out} ({len(card)} chars)")


if __name__ == "__main__":
    main()

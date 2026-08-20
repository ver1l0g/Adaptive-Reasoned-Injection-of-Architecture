#!/usr/bin/env python3
"""analyze_graphs.py 鈥?aggregate structural analysis of gpnn runs.

Two input modes:
  1. Dump files (stdout logs containing "[Graph]" lines):
       python analyze_graphs.py run1.log run2.log ...
     Aggregates node-type distribution, depth, params across runs.
  2. Expression CSVs (e.g. results/feynman_results.csv with an expression column):
       python analyze_graphs.py --expr results/feynman_results.csv
     Counts operator frequencies in recovered formulas.
"""
import csv, re, sys
from collections import Counter

def parse_dumps(paths):
    node_re = re.compile(r"\[Graph\] nodes=(\d+) edges=(\d+) params=(\d+) depth=(\d+)")
    types_re = re.compile(r"\[Graph\] types:(.*)")
    type_kv = re.compile(r"(\w+)=(\d+)")
    runs = 0
    agg = Counter()
    depths, params_list, nodes_list = [], [], []
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"skip {p}: {e}", file=sys.stderr); continue
        for m in node_re.finditer(text):
            nodes_list.append(int(m.group(1)))
            params_list.append(int(m.group(3)))
            depths.append(int(m.group(4)))
            runs += 1
            tm = types_re.search(text[m.end():m.end()+600])
            if tm:
                for tname, cnt in type_kv.findall(tm.group(1)):
                    agg[tname] += int(cnt)
    if runs == 0:
        print("no [Graph] lines found"); return
    print(f"runs analyzed: {runs}")
    print(f"nodes:  mean={sum(nodes_list)/runs:.1f} max={max(nodes_list)}")
    print(f"params: mean={sum(params_list)/runs:.0f} max={max(params_list)}")
    print(f"depth:  mean={sum(depths)/runs:.1f} max={max(depths)}")
    total = sum(agg.values())
    print(f"\nnode-type distribution (all runs, {total} nodes):")
    for tname, cnt in agg.most_common():
        bar = "#" * max(1, int(40 * cnt / total))
        print(f"  {tname:<10} {cnt:>6}  {100*cnt/total:5.1f}% {bar}")

def parse_expr_csv(path):
    ops = Counter()
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e = row.get("expression") or row.get("expr") or ""
            if not e: continue
            n += 1
            for pat, name in [(r"tanh\(", "tanh"), (r"sin\(", "sin"),
                              (r"\*", "multiply"), (r"/", "divide"),
                              (r"\+", "add"), (r"-", "subtract"),
                              (r"\?1:0", "ifelse/cond"), (r"!=", "xor/ne"),
                              (r">", "greater")]:
                ops[name] += len(re.findall(pat, e))
    print(f"expressions analyzed: {n}")
    total = sum(ops.values())
    if total:
        print(f"operator occurrences ({total} total):")
        for name, cnt in ops.most_common():
            bar = "#" * max(1, int(40 * cnt / total))
            print(f"  {name:<12} {cnt:>6}  {100*cnt/total:5.1f}% {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "--expr":
        parse_expr_csv(sys.argv[2])
    else:
        parse_dumps(sys.argv[1:])

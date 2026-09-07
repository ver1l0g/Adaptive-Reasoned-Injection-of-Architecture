#!/usr/bin/env python3
"""Merge run-local subgraph libraries into the central library.
Dedup by canonical_expression. Usage (from problems/ or repo root):
  python harness/merge_libraries.py <central_path> [search_root]
Defaults: central = ../library/central_subgraph.txt, search = ./
"""
import os, sys

def parse_lib(path):
    entries = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            n = int(f.readline().strip())
            for _ in range(n):
                header = f.readline().rstrip("\n")
                fp_line = f.readline().rstrip("\n")
                # M7.7 arch line (optional, starts with a quote)
                save = f.tell()
                nxt = f.readline().rstrip("\n")
                arch = ""
                if nxt.startswith('"'):
                    arch = nxt
                else:
                    f.seek(save)
                entries.append((header, fp_line, arch))
    except (OSError, ValueError):
        pass
    return entries

def canon(header):
    # 3rd tab-separated quoted field = canonical expression
    parts = header.split("\t")
    return parts[2] if len(parts) >= 3 else header

central = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "library", "central_subgraph.txt")
root = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(os.path.dirname(central), exist_ok=True)

merged = {}
for dirpath, _, files in os.walk(root):
    if "archives" in dirpath or ".git" in dirpath:
        continue
    for fn in files:
        if fn == "subgraph_library.txt":
            p = os.path.join(dirpath, fn)
            for e in parse_lib(p):
                key = canon(e[0])
                if key and key not in merged:
                    merged[key] = e

with open(central, "w", encoding="utf-8") as f:
    f.write(f"{len(merged)}\n")
    for e in merged.values():
        f.write(e[0] + "\n" + e[1] + ("\n" + e[2] + "\n" if e[2] else "\n"))
print(f"central library: {len(merged)} unique entries -> {central}")

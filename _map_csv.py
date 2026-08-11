#!/usr/bin/env python3
"""Map every .csv file to its producer(s) and consumer(s) across scripts."""
import ast, re
from pathlib import Path
from collections import defaultdict

DATA = Path(".")

# 1. For each script, collect: constants mapping name->".csv" path, to_csv calls, read_csv calls
producers = defaultdict(list)   # csvfile -> [(script, linenum)]
consumers = defaultdict(list)   # csvfile -> [(script, linenum)]
script_consts = {}              # script -> {CONSTNAME: "file.csv"}

for py in sorted(DATA.glob("*.py")):
    src = py.read_text(encoding="utf-8", errors="ignore")
    # collect string literals in module that end in .csv or are file paths
    consts = {}
    # assignments like X = "name.csv" or X = DATA_DIR/"name.csv"
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*[^\n]*?["\']([^"\']*\.csv)["\']', src, re.M):
        consts[m.group(1)] = m.group(2)
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*[^\n]*?/\s*["\']([^"\']*\.csv)["\']', src, re.M):
        consts[m.group(1)] = m.group(2)
    script_consts[py.stem] = consts

    for m in re.finditer(r'(?:\.to_csv|\.to_parquet)\(([^)]*)\)', src):
        pass
    # to_csv calls: resolve arg
    for m in re.finditer(r'\.to_csv\(([^\n()]*)\)', src):
        arg = m.group(1).strip()
        fname = None
        mm = re.match(r'DATA_DIR\s*/\s*["\']([^"\']+\.csv)["\']', arg)
        if mm: fname = mm.group(1)
        else:
            mm2 = re.match(r'["\']([^"\']+\.csv)["\']', arg)
            if mm2: fname = mm2.group(1)
            else:
                # constant ref
                cm = re.match(r'([A-Z][A-Z0-9_]*)', arg)
                if cm and cm.group(1) in consts: fname = consts[cm.group(1)]
        if fname:
            producers[fname].append((py.stem, m.start()))
    for m in re.finditer(r'\.read_csv\(([^\n()]*)\)', src):
        arg = m.group(1).strip()
        fname = None
        mm = re.match(r'DATA_DIR\s*/\s*["\']([^"\']+\.csv)["\']', arg)
        if mm: fname = mm.group(1)
        else:
            mm2 = re.match(r'["\']([^"\']+\.csv)["\']', arg)
            if mm2: fname = mm2.group(1)
            else:
                cm = re.match(r'([A-Z][A-Z0-9_]*)', arg)
                if cm and cm.group(1) in consts: fname = consts[cm.group(1)]
        if fname:
            consumers[fname].append((py.stem, m.start()))

# Also match OUT = DATA_DIR / "name.csv" style constants
for py in sorted(DATA.glob("*.py")):
    src = py.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r'^\s*(OUT\w*|[A-Z][A-Z0-9_]*)\s*=\s*DATA_DIR\s*/\s*["\']([^"\']+\.csv)["\']', src, re.M):
        pass

all_files = sorted(set(producers) | set(consumers))
print(f"=== CSV files referenced in scripts: {len(all_files)} ===")
print("\n=== PRODUCED ONLY (derived, safe to convert to parquet) ===")
for f in all_files:
    if f in producers and f not in consumers:
        print(f"  {f}  <- {[p[0] for p in producers[f]]}")
print("\n=== BOTH producer and consumer (must convert both) ===")
for f in all_files:
    if f in producers and f in consumers:
        print(f"  {f}  P:{[p[0] for p in producers[f]]}  C:{[c[0] for c in consumers[f]]}")
print("\n=== CONSUMED ONLY (source inputs, KEEP as csv) ===")
for f in all_files:
    if f not in producers and f in consumers:
        print(f"  {f}  <- {[c[0] for c in consumers[f]]}")
print("\n=== CSV files on disk NOT referenced by any script ===")
disk = {p.name for p in DATA.glob("*.csv")}
refd = set(all_files)
for f in sorted(disk - refd):
    print(f"  {f}  (no script ref)")

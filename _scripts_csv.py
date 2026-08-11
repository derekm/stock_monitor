#!/usr/bin/env python3
"""Generate the unique list of scripts that touch any CSV (produce or consume)."""
import ast, re
from pathlib import Path

DATA = Path(".")
scripts = set()
consts = {}
for py in DATA.glob("*.py"):
    src = py.read_text(encoding="utf-8", errors="ignore")
    if ".to_csv(" not in src and ".read_csv(" not in src:
        continue
    if any(t in py.name for t in ("_map_csv", "_fix", "test_", "backfill_historical", "parse_sp500", "sp_index", "sp_history", "sp_universe")):
        continue
    scripts.add(py.name)

for s in sorted(scripts):
    print(s)
print("TOTAL", len(scripts))

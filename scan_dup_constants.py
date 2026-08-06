#!/usr/bin/env python3
"""Find numeric literals that appear in 3+ files as inline values (not via a
shared constant) — candidate magic-number duplication. Only flags floats in
comparison/assignment contexts, excluding obvious loop/count/date values.
"""
import ast, collections, os, re

ROOT = r"C:/Users/derek/src/stockmagic/stock_monitor"
FLOAT_RE = re.compile(r"^[0-9]+\.[0-9]+$")

def main():
    by_val = collections.defaultdict(set)  # value -> set of files
    for fn in os.listdir(ROOT):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(ROOT, fn)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            # comparisons like x >= 0.15, x <= 9.0
            if isinstance(node, ast.Compare):
                for c in node.comparators:
                    if isinstance(c, ast.Constant) and isinstance(c.value, float):
                        by_val[repr(c.value)].add(fn)
            # assign to non-constant name
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, float):
                for t in node.targets:
                    if isinstance(t, ast.Name) and not t.id.isupper():
                        by_val[repr(node.value.value)].add(fn)
    print("== float literals in 3+ files (magic-number dupes) ==")
    for v, files in sorted(by_val.items()):
        if len(files) >= 3:
            print(f"{v}: {', '.join(sorted(files))}")

if __name__ == "__main__":
    main()

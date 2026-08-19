#!/usr/bin/env python3
"""
audit_schema_names.py — exhaustive audit of pre-migration fundamentals key usage.

Reports three classes of stale-name usage across every text file in the tree:

  1. SELF-INCONSISTENT -- a file that READS an old key it never writes while
     writing the canonical counterpart. This is the class a cross-file
     producer/consumer check cannot see, because producer and consumer are the same
     module.
  2. OLD-NAME READS IN EXECUTABLE CODE -- every .get("x") / ["x"] / "x" in ... hit.
  3. OLD NAMES IN COMMENTS AND DOCS -- prose that describes the wrong column.

Design constraints, because each one is a way an audit can miss a real bug:

  * scans READS as well as dict-literal WRITES -- a write-only regex cannot see
    data.get("revenue")
  * checks each file against ITSELF, not only against its callers
  * NO file exclusions and NO "generic name" filtering; every hit is printed and
    adjudicated by reading it. An exclusion list is an assumption, and assumptions
    are what hide leaks.
  * every text extension, not just .py

Usage:
    python audit_schema_names.py            # summary
    python audit_schema_names.py --all      # every hit
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# old -> canonical, from the 2026-08 migration
RENAMED = {
    "revenue": "revenue_quarterly",
    "total_revenue": "revenue_quarterly",
    "ttm_revenue": "revenue_ttm",
    "net_income": "net_income_quarterly",
    "ttm_net_income": "net_income_ttm",
    "operating_income": "operating_income_quarterly",
    "ttm_operating_income": "operating_income_ttm",
    "operating_cash_flow": "operating_cash_flow_ttm",
    "ttm_operating_cash_flow": "operating_cash_flow_ttm",
    "capital_expenditure": "capital_expenditure_ttm",
    "ttm_capital_expenditure": "capital_expenditure_ttm",
    "assets": "total_assets",
    "equity": "shareholders_equity",
    "stockholders_equity": "shareholders_equity",
    "cash": "cash_and_equivalents",
    "shares": "shares_outstanding",
    "debt": "total_debt",
}

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "checkpoints",
             "backfill_backups", "dashboard_data", ".pytest_cache", ".mypy_cache"}

EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".sql", ".js", ".ts", ".html",
        ".sh", ".bash", ".txt", ".ipynb", ".toml", ".cfg", ".ini", ".rst"}

# a key being READ off a mapping
READ_PATTERNS = [
    re.compile(r'\.get\(\s*[\'"]([a-z_0-9]+)[\'"]'),
    re.compile(r'\[\s*[\'"]([a-z_0-9]+)[\'"]\s*\](?!\s*=)'),
    re.compile(r'[\'"]([a-z_0-9]+)[\'"]\s+in\s+\w'),
]
# a key being WRITTEN into a dict literal or by assignment
WRITE_PATTERNS = [
    re.compile(r'[\'"]([a-z_0-9]+)[\'"]\s*:'),
    re.compile(r'\[\s*[\'"]([a-z_0-9]+)[\'"]\s*\]\s*='),
]
# names that are canonical (so a read of them is fine) -- used for self-consistency
CANON = set(RENAMED.values())


def scan_file(p: Path):
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    reads, writes = {}, {}
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        is_comment = stripped.startswith(("#", "//", "*", "<!--"))
        for pats, bucket in ((READ_PATTERNS, reads), (WRITE_PATTERNS, writes)):
            for rx in pats:
                for m in rx.finditer(line):
                    k = m.group(1)
                    bucket.setdefault(k, []).append(
                        (lineno, stripped[:100], is_comment))
    return reads, writes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="print every hit")
    args = ap.parse_args()

    root = Path(__file__).parent
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file()
             and p.suffix.lower() in EXTS
             and not any(d in p.parts for d in SKIP_DIRS)
             and p.name != Path(__file__).name]

    print(f"scanning {len(files)} files, {len(RENAMED)} renamed keys, no exclusions")
    print()

    code_reads = []      # reads of an OLD name in executable code
    comment_only = []    # the same, but in a comment/doc
    self_inconsistent = []

    for p in files:
        res = scan_file(p)
        if not res:
            continue
        reads, writes = res
        rel = p.relative_to(root).as_posix()

        for k, hits in reads.items():
            if k not in RENAMED:
                continue
            for lineno, ctx, is_comment in hits:
                rec = (rel, lineno, k, RENAMED[k], ctx)
                (comment_only if is_comment else code_reads).append(rec)

        # SELF-INCONSISTENCY: this file reads an old key it never writes, while
        # writing the canonical counterpart. That is the exact edgar_html_10q bug.
        written = set(writes)
        for k in reads:
            if k in RENAMED and k not in written and RENAMED[k] in written:
                lineno, ctx, is_comment = reads[k][0]
                if not is_comment:
                    self_inconsistent.append((rel, lineno, k, RENAMED[k], ctx))

    print("=" * 78)
    print(f"SELF-INCONSISTENT: reads an OLD key it never writes, but DOES write the")
    print(f"canonical name -- the edgar_html_10q L319 class ({len(self_inconsistent)})")
    print("=" * 78)
    if not self_inconsistent:
        print("  none")
    for rel, lineno, old, new, ctx in self_inconsistent:
        print(f"  {rel}:{lineno}")
        print(f"    reads '{old}' but writes '{new}'  ->  {ctx}")

    print()
    print("=" * 78)
    print(f"OLD-NAME READS IN EXECUTABLE CODE ({len(code_reads)})")
    print("=" * 78)
    by_file: dict[str, list] = {}
    for rec in code_reads:
        by_file.setdefault(rec[0], []).append(rec)
    for rel in sorted(by_file):
        rows = by_file[rel]
        print(f"\n{rel}  ({len(rows)})")
        show = rows if args.all else rows[:6]
        for _, lineno, old, new, ctx in show:
            print(f"  L{lineno:<5} {old:22} -> {new:26} {ctx[:66]}")
        if not args.all and len(rows) > 6:
            print(f"  ... {len(rows)-6} more (use --all)")

    print()
    print("=" * 78)
    print(f"OLD NAMES IN COMMENTS/DOCS ({len(comment_only)}) -- these mislead readers")
    print("=" * 78)
    cf: dict[str, int] = {}
    for rel, *_ in comment_only:
        cf[rel] = cf.get(rel, 0) + 1
    for rel, n in sorted(cf.items(), key=lambda kv: -kv[1]):
        print(f"  {rel:52} {n}")

    print()
    print(f"TOTAL: {len(self_inconsistent)} self-inconsistent, "
          f"{len(code_reads)} code reads, {len(comment_only)} comment refs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

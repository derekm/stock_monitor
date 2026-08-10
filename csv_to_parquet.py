#!/usr/bin/env python3
"""csv_to_parquet.py — convert derived CSV outputs to parquet.

Why: 318MB of CSVs (subindustry_regime.parquet alone = 239MB) served to the
DuckDB-Wasm dashboard. Parquet is columnar + compressed: 239MB → ~30-60MB,
and DuckDB reads parquet natively with schema (no CSV type guessing).

DATE-NATIVE convention: any column whose NAME looks like a date key
(date/as_of/ref_date/end_date/..._date) OR whose values parse as YYYY-MM-DD
is stored as pyarrow DATE (date32[day]) — never datetime64/midnight
timestamp. That keeps the repo's "daily date-key columns stay DATE" rule.

Rules:
  - Only converts .csv (never touches canonical parquet tables)
  - Skips if a parquet twin exists AND is newer than the csv
  - Never deletes the CSV (pandas readers still use them)
  - Writes <stem>.parquet next to the CSV

Usage: python csv_to_parquet.py [--min-size 1048576] [--all]
  --min-size: only convert CSVs >= this many bytes (default 1MB)
  --all: convert every CSV regardless of size
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SKIP = {".git", "node_modules", "logs", "__pycache__", "dashboard_data", "checkpoints", ".venv"}

DATE_NAME_RE = re.compile(
    r"(^|_)(date|as_of|asof|ref_date|end_date|start_date|filed_at|buy_date|sell_date|"
    r"added_date|valid_from|valid_to|observation_date|period_end|ex_date|pay_date|"
    r"trade_date|report_date)(_|$)"
)
DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def _looks_like_date(name: str) -> bool:
    n = str(name).lower().strip()
    return bool(DATE_NAME_RE.search(n)) or n in ("date", "as_of_date", "ref_date")


def _is_date_series(s: pd.Series) -> bool:
    """True if >80% of non-null string values parse as YYYY-MM-DD."""
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    ok = sample.map(lambda v: bool(DATE_VALUE_RE.match(v.strip()))).mean()
    return ok > 0.8


def convert_csv(path: Path, force: bool = False) -> tuple[str, str]:
    """Convert one CSV to parquet (DATE-native). Returns (status, detail)."""
    out = path.with_suffix(".parquet")
    if out.exists() and not force and out.stat().st_mtime >= path.stat().st_mtime:
        return "skip", f"{path.name} (parquet newer)"
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return "fail", f"{path.name}: read {e}"
    if df.empty:
        return "skip", f"{path.name} (empty)"

    # date-native: convert date-like columns to datetime.date objects
    n_date = 0
    for c in df.columns:
        if _looks_like_date(c) or _is_date_series(df[c]):
            try:
                dt = pd.to_datetime(df[c], errors="coerce")
                # keep date32 semantics: map to python date objects
                df[c] = dt.dt.date.where(dt.notna())
                n_date += 1
            except Exception:
                pass
    try:
        df.to_parquet(out, index=False, engine="pyarrow")
    except Exception as e:
        return "fail", f"{path.name}: write {e}"

    before = path.stat().st_size
    after = out.stat().st_size
    pct = 100 * (1 - after / before) if before else 0
    return "ok", (
        f"{path.name}: {before/1048576:.1f}MB → {after/1048576:.1f}MB "
        f"(-{pct:.0f}%) · {len(df)} rows · {len(df.columns)} cols · {n_date} date cols"
    )


def main(min_size: int, all_files: bool, force: bool) -> None:
    csvs = []
    for p in DATA_DIR.rglob("*.csv"):
        if any(part in SKIP for part in p.parts):
            continue
        if p.name.startswith("_"):
            continue
        if not all_files and p.stat().st_size < min_size:
            continue
        csvs.append(p)
    csvs.sort()
    print(f"csv_to_parquet: {len(csvs)} CSVs to consider (min {min_size/1048576:.0f}MB, all={all_files})")

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for p in csvs:
        status, detail = convert_csv(p, force=force)
        stats[status] += 1
        print(f"  [{status:>4}] {detail}")
    print(f"\nDone: {stats['ok']} converted, {stats['skip']} skipped, {stats['fail']} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=1024 * 1024, help="min bytes (default 1MB)")
    ap.add_argument("--all", action="store_true", help="convert every CSV")
    ap.add_argument("--force", action="store_true", help="overwrite existing parquet twins")
    args = ap.parse_args()
    main(args.min_size, args.all, args.force)

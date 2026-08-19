#!/usr/bin/env python3
"""
verify_force_pass.py — prove a re-run actually rewrote rows written by an older
version of the extractor.

Diffs the current panel against a pre-run backup on (ticker, as_of_date) for the
rows that backup marks source == edgar_v2, and reports how many values moved in each
column. A re-run that skips those rows exits 0 and reports success just like one that
rewrites them, so the rewrite has to be measured rather than assumed.

Stale rows are not necessarily wrong -- where two candidate XBRL tags agree closely
an old value can match the correct one exactly -- so the question is which values
CHANGED, not whether they look plausible.

The pre-run backup (backfill_backups/fundamentals_pre_full_v2_*.parquet) is the
authoritative source of the old VALUES; _old_v2_tickers.txt is only a convenience
list for feeding --tickers.

USAGE
    python verify_force_pass.py --compare
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
BACKUP_GLOB = str(DATA_DIR / "backfill_backups" / "fundamentals_pre_full_v2_*.parquet")

# columns the extractor fixes actually affect
COLS = [
    "revenue_ttm", "net_income_ttm", "operating_income_ttm",
    "operating_cash_flow_ttm", "capital_expenditure_ttm",
    "revenue_quarterly", "net_income_quarterly",
    "total_assets", "shareholders_equity", "total_debt",
    "cash_and_equivalents", "shares_outstanding",
    "free_cash_flow", "fcf_margin", "roe", "roic",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    backups = sorted(glob.glob(BACKUP_GLOB))
    if not backups:
        print(f"no pre-run backup matching {BACKUP_GLOB}")
        return 1
    bk = backups[-1]
    print(f"pre-force values from : {Path(bk).name}")
    print(f"current panel         : {FUND.name}")

    b = pl.read_parquet(bk)
    f = pl.read_parquet(FUND)

    # The old-extractor rows identify themselves in the backup: source == edgar_v2.
    old = b.filter(pl.col("source") == "edgar_v2")
    print()
    print(f"old-extractor rows: {old.height:,} across "
          f"{old['ticker'].n_unique()} tickers")

    key = ["ticker", "as_of_date"]
    cols = [c for c in COLS if c in old.columns and c in f.columns]
    j = old.select(key + cols).join(
        f.select(key + cols), on=key, how="inner", suffix="_new")
    print(f"matched in current panel: {j.height:,}")

    if not args.compare:
        print()
        print("pass --compare to diff old vs current")
        return 0

    print()
    print(f"{'column':28} {'rows changed':>13} {'now non-null':>14}")
    total_changed = 0
    for c in cols:
        n = c + "_new"
        # a row counts as changed if the value moved, appeared, or disappeared
        diff = j.filter(
            ((pl.col(c).is_not_null() & pl.col(n).is_not_null())
             & ((pl.col(c) - pl.col(n)).abs() > 1e-6))
            | (pl.col(c).is_null() & pl.col(n).is_not_null())
            | (pl.col(c).is_not_null() & pl.col(n).is_null())
        )
        total_changed += diff.height
        filled = j.filter(pl.col(c).is_null() & pl.col(n).is_not_null()).height
        emptied = j.filter(pl.col(c).is_not_null() & pl.col(n).is_null()).height
        note = ""
        if filled:
            note += f" +{filled} filled"
        if emptied:
            note += f" -{emptied} emptied"
        print(f"  {c:26} {diff.height:>13,} {j[n].drop_nulls().len():>14,}{note}")

    print()
    # 3 stray changes are noise (a row gaining a value), not a force pass. A real
    # force pass rewrites thousands of values across many columns, so require a
    # meaningful fraction of the matched rows to have moved before calling it done.
    cols_moved = sum(
        1 for c in cols
        if j.filter(
            ((pl.col(c).is_not_null() & pl.col(c + "_new").is_not_null())
             & ((pl.col(c) - pl.col(c + "_new")).abs() > 1e-6))
            | (pl.col(c).is_null() & pl.col(c + "_new").is_not_null())
            | (pl.col(c).is_not_null() & pl.col(c + "_new").is_null())
        ).height > 0
    )
    took_effect = total_changed >= max(50, j.height * 0.01) and cols_moved >= 3
    print(f"values changed: {total_changed:,}   columns touched: {cols_moved}/{len(cols)}")
    if took_effect:
        print("re-run DID rewrite the old rows ✓")
    else:
        print("re-run did NOT rewrite the old rows -- they still hold the old "
              "extractor's values. Check that source rank permits the overwrite "
              "(update_fundamentals.SOURCE_RANK: incoming must be >= stored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

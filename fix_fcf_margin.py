#!/usr/bin/env python3
"""
fix_fcf_margin.py — repair fcf_margin values that cannot be true.

fcf_margin = free_cash_flow / revenue. Both sides must share a period basis:
free_cash_flow is TTM (operating_cash_flow_ttm - |capital_expenditure_ttm|), so the
denominator is revenue_ttm.

Two causes produce an impossible margin, and only one is recomputable:

  * the row has free_cash_flow AND revenue_ttm -> RECOMPUTE from the TTM pair
  * the row has NO revenue at all (revenue_ttm and revenue_quarterly both NULL) yet
    still carries a numeric fcf_margin. Nothing can be divided, so the stored number
    cannot be reconstructed or trusted -> NULL

A wrong number is worse than a gap: fcf_margin feeds Damodaran quality screens and
preferred_metrics scoring, where an extreme value dominates any ranking. An explicit
NULL is honest and every consumer already handles it.

CAUTION: an impossible margin can be a SYMPTOM of a wrong denominator rather than a
wrong margin. Where revenue_ttm is understated the margin is arithmetically correct
for its inputs, and nulling it hides the real defect. Check the denominator before
running this.

Usage:
    python fix_fcf_margin.py --dry-run
    python fix_fcf_margin.py --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
BACKUP_DIR = DATA_DIR / "backfill_backups"

# A real business does not sustain FCF above revenue. Keep the threshold loose so
# only clearly impossible values are touched; a genuine one-off (asset sale, tax
# refund) can legitimately push a single quarter above 1.0.
IMPOSSIBLE = 1.5


def plan(df: pd.DataFrame) -> dict:
    bad = df["fcf_margin"].notna() & (df["fcf_margin"].abs() > IMPOSSIBLE)
    has_ttm = df["free_cash_flow"].notna() & df["revenue_ttm"].notna() & (df["revenue_ttm"] > 0)
    return {
        "bad": int(bad.sum()),
        "recompute": int((bad & has_ttm).sum()),
        "null": int((bad & ~has_ttm).sum()),
        "bad_mask": bad,
        "recompute_mask": bad & has_ttm,
        "null_mask": bad & ~has_ttm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(FUND)
    print(f"{FUND.name}: {len(df):,} rows")
    print(f"fcf_margin non-null: {int(df['fcf_margin'].notna().sum()):,}")
    print(f"  min {df['fcf_margin'].min():,.1f}   max {df['fcf_margin'].max():,.1f}")

    p = plan(df)
    print()
    print(f"|fcf_margin| > {IMPOSSIBLE}: {p['bad']} rows")
    print(f"  recomputable from revenue_ttm : {p['recompute']}")
    print(f"  no revenue to divide by -> NULL: {p['null']}")

    if p["bad"] == 0:
        print("nothing to fix ✓")
        return 0

    sample = df.loc[p["recompute_mask"], ["ticker", "as_of_date", "fcf_margin",
                                          "free_cash_flow", "revenue_ttm"]].head(6)
    if len(sample):
        print()
        print("sample recomputations:")
        for r in sample.itertuples():
            new = r.free_cash_flow / r.revenue_ttm
            print(f"  {r.ticker:8} {str(r.as_of_date)[:10]}  {r.fcf_margin:12,.2f} -> {new:7.4f}")

    if not args.apply:
        print()
        print("dry run -- nothing written. Re-run with --apply.")
        return 0

    before_nn = int(df["fcf_margin"].notna().sum())
    df.loc[p["recompute_mask"], "fcf_margin"] = (
        df.loc[p["recompute_mask"], "free_cash_flow"]
        / df.loc[p["recompute_mask"], "revenue_ttm"]
    )
    df.loc[p["null_mask"], "fcf_margin"] = np.nan
    after_nn = int(df["fcf_margin"].notna().sum())

    assert before_nn - after_nn == p["null"], (
        f"expected {p['null']} values nulled, got {before_nn - after_nn}")
    left = int((df["fcf_margin"].notna() & (df["fcf_margin"].abs() > IMPOSSIBLE)).sum())
    assert left == 0, f"{left} impossible values remain after the fix"

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = BACKUP_DIR / f"fundamentals_pre_fcfmargin_{stamp}.parquet"
    shutil.copy2(FUND, bk)

    tmp = FUND.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, FUND)

    print()
    print(f"backup      : {bk.name}")
    print(f"recomputed  : {p['recompute']}")
    print(f"nulled      : {p['null']}")
    print(f"fcf_margin non-null: {before_nn:,} -> {after_nn:,}")
    print(f"rows unchanged: {len(df):,}")
    print(f"new range: {df['fcf_margin'].min():.4f} .. {df['fcf_margin'].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

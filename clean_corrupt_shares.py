#!/usr/bin/env python3
"""
clean_corrupt_shares.py — repair implausible shares_outstanding in fundamentals.parquet.

WHY A SIMPLE THRESHOLD DOES NOT WORK

The obvious rule ("NULL anything above 1e11") destroys real data. Measured on this
panel:

  HCMC   shares_outstanding 5.27e11, market_cap 5.27e6 -> implied price $0.00001
         That is CORRECT: HCMC is a sub-penny stock that really has ~380B shares
         outstanding, and its series grows monotonically through repeated dilution.
         A 1e11 cutoff would delete 20 legitimate rows.

Nor does "ratio to the ticker's own median": the 20x-100x band is full of genuine
corporate actions --

  SCLX 2022-12-31  7.03e6 -> 1.41e8   (real issuance,   20.1x median)
  AERA 2024-02-29  1.16e8 -> 2.33e9   (real issuance,   20.2x median)
  VYST 2021-09-30  6.39e7 -> 1.29e9   (real issuance,   20.3x median)

WHAT ACTUALLY IDENTIFIES CORRUPTION

A share count is a slow-moving stock quantity. Real changes persist: after a split
or an issuance the new level STAYS. Corruption is a value that is wildly out of
line with the ticker's own neighbours and then reverts -- a unit/scale error in one
filing, not a corporate action:

  AAQL 2017-09-30   9.99e6 -> 9.99e10 -> 9.99e6      (exactly 1e4 too big)
  AATC 2016-06-30   5.04e6 -> 5.05e9  -> 5.07e6      (exactly 1e3 too big)
  ADAM 2011-12-31   1.12e7 -> 1.39e10 -> 1.42e7
  CNA  2012-12-31   2.71e8 -> 2.69e14 (sustained run, then back to 2.7e8)

So the rule is: flag a row when it exceeds a robust per-ticker baseline (the median
of the OTHER rows for that ticker) by a large factor AND the series returns to that
baseline afterwards. Sustained runs are caught by comparing against the median
rather than only the immediate neighbours.

REPAIR, NOT DELETION

Where the corrupt value is a clean power-of-ten multiple of the baseline (1e3, 1e4,
1e6 ...), it is a unit error and is rescaled. Otherwise the value is set to NULL --
an honest gap beats a wrong number. Nothing is dropped; only shares_outstanding is
touched.

Usage:
    python clean_corrupt_shares.py --dry-run
    python clean_corrupt_shares.py --apply
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

COL = "shares_outstanding"
SPIKE_FACTOR = 50.0      # vs the median of the ticker's OTHER rows
REVERT_FACTOR = 10.0     # the series must come back down to within this of baseline
MIN_ROWS = 4             # need enough history for a meaningful baseline
PLACEHOLDER_MAX = 1000.0  # 1.0 / 4.0 / 20.0 etc are filing placeholders, not counts


def _px_plausible(px: float) -> bool:
    """Is an implied share price (market_cap / shares) believable?

    Deliberately WIDE: this is only used to break a tie about which of two
    power-of-ten candidates is right, never on its own to condemn a row. It must
    not be used as primary evidence, because market_cap itself is unreliable --
    HCMC's early rows imply a $20,000,000 share price, which means those market
    caps are wrong, not the share counts.
    """
    return np.isfinite(px) and 1e-6 <= px <= 1e7


def find_corrupt(df: pd.DataFrame) -> pd.DataFrame:
    """Return the flagged rows with a proposed repair for each."""
    out = []
    cols = ["ticker", "as_of_date", COL, "source"]
    if "market_cap" in df.columns:
        cols.append("market_cap")
    d = df[cols].copy()
    if "market_cap" not in d.columns:
        d["market_cap"] = np.nan
    d = d[d[COL].notna()]

    for ticker, g in d.groupby("ticker", sort=False):
        if len(g) < MIN_ROWS:
            continue
        g = g.sort_values("as_of_date")
        vals = g[COL].to_numpy(dtype=float)
        mcaps = g["market_cap"].to_numpy(dtype=float)
        pos = vals > 0
        if pos.sum() < MIN_ROWS:
            continue

        for i in range(len(g)):
            v = vals[i]
            if not np.isfinite(v) or v <= 0:
                continue
            others = np.delete(vals, i)
            others = others[(others > 0) & np.isfinite(others)]
            if len(others) < MIN_ROWS - 1:
                continue
            base = float(np.median(others))
            if base <= 0:
                continue

            mc = mcaps[i]

            # PRIMARY evidence is the ticker's own series only. market_cap is NOT
            # used to condemn a row -- it is unreliable in exactly the same rows
            # (HCMC's early market caps imply a $2e7 share price).
            too_big = v / base >= SPIKE_FACTOR
            # A handful of filings carry a placeholder instead of a share count
            # (1.0, 4.0, 20.0 ...). Those are not scale errors and cannot be
            # rescaled -- 275 rows sit at <=100 shares. Only treat a small value
            # as corrupt when it is an absurd absolute count, not merely small
            # relative to a polluted median.
            placeholder = v <= PLACEHOLDER_MAX and base >= SPIKE_FACTOR * v

            if not (too_big or placeholder):
                continue

            if too_big:
                later = vals[i + 1:]
                later = later[(later > 0) & np.isfinite(later)]
                if len(later) and np.median(later) > base * REVERT_FACTOR:
                    continue      # level persisted -> genuine corporate action

            ratio = v / base if too_big else base / v
            # A power-of-ten unit error can go EITHER WAY, so the median is not
            # automatically the truth. APG's series is 170, 169, 1.69e8, 1.70e8 --
            # the median (8.45e7) is fine but the SMALL values are the corrupt
            # ones, and WMG's median (1.055e3) is itself corrupt. Anchor on
            # market_cap when available: implied price = market_cap / shares must
            # be plausible ($0.00001-$1e6 covers sub-penny stocks through BRK-A).
            exp = round(np.log10(ratio))
            repair = np.nan
            kind = "null"
            # Placeholders (1.0, 20.0) are not scale errors -- there is no true
            # value hiding behind them, so they become NULL.
            if too_big and exp >= 2 and abs(ratio / (10.0 ** exp) - 1.0) < 0.15:
                cand = v / (10.0 ** exp)
                ok = True
                if np.isfinite(mc) and mc > 0:
                    ok = _px_plausible(mc / cand)
                if ok:
                    repair = cand
                    kind = f"rescale_1e{int(exp)}"
            out.append({
                "ticker": ticker,
                "as_of_date": g["as_of_date"].iloc[i],
                "bad_value": v,
                "baseline": base,
                "ratio": ratio,
                "repair": repair,
                "action": kind,
                "source": g["source"].iloc[i],
            })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(FUND)
    print(f"{FUND.name}: {len(df):,} rows")
    print(f"{COL} non-null: {int(df[COL].notna().sum()):,}")

    bad = find_corrupt(df)
    if bad.empty:
        print("no corrupt share counts found ✓")
        return 0

    print()
    print(f"flagged {len(bad)} rows across {bad['ticker'].nunique()} tickers")
    print(bad["action"].value_counts().to_string())
    print()
    print("worst 12 by ratio:")
    show = bad.sort_values("ratio", ascending=False).head(12)
    for r in show.itertuples():
        rep = "NULL" if not np.isfinite(r.repair) else f"{r.repair:.4e}"
        print(f"  {r.ticker:8} {str(r.as_of_date):12} {r.bad_value:.4e} "
              f"-> {rep:12} (baseline {r.baseline:.3e}, {r.ratio:.0f}x, {r.action})")

    # explicit sanity assertions: legitimate high-share-count names must be spared
    for t in ("HCMC",):
        if t in set(bad["ticker"]):
            print()
            print(f"ABORT: {t} was flagged, but its high share count is real "
                  "(sub-penny stock, monotonic dilution). Detector is too loose.")
            return 1

    if not args.apply:
        print()
        print("dry run -- nothing written. Re-run with --apply.")
        return 0

    key = df["ticker"].astype(str) + "|" + df["as_of_date"].astype(str)
    bad_key = bad["ticker"].astype(str) + "|" + bad["as_of_date"].astype(str)
    repair_map = dict(zip(bad_key, bad["repair"]))

    before_nn = int(df[COL].notna().sum())
    mask = key.isin(set(bad_key))
    df.loc[mask, COL] = key[mask].map(repair_map)
    after_nn = int(df[COL].notna().sum())

    n_null = int(bad["repair"].isna().sum())
    n_fix = len(bad) - n_null
    assert before_nn - after_nn == n_null, (
        f"expected {n_null} values to become NULL, got {before_nn - after_nn}")

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = BACKUP_DIR / f"fundamentals_pre_sharesclean_{stamp}.parquet"
    shutil.copy2(FUND, bk)

    tmp = FUND.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, FUND)

    print()
    print(f"backup  : {bk.name}")
    print(f"rescaled: {n_fix}   nulled: {n_null}")
    print(f"{COL} non-null: {before_nn:,} -> {after_nn:,}")
    print(f"rows unchanged: {len(df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

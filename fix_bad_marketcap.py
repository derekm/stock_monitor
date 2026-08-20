#!/usr/bin/env python3
"""
Null out absurd market_cap / market_cap_b rows in fundamentals.parquet.

Bad = unit errors from EDGAR XBRL ingestion:
  - market_cap_b > 100,000  (1e6x too big, e.g. FITB $5.87T, HII $1.58T)
  - market_cap_b > 0 and < 0.05  (1e3x too small, e.g. RTX $30M, AAPL $14M)

Both market_cap and market_cap_b carry the same error together (ratio check
cannot catch them), so we null the pair on the same rows. Downstream
(add_daily_marketcap, implied_r_screen, sp_index_methodology) already handle
NaN by skipping/gapping, so nulling is the safe repair.

This module also exports `bad_marketcap_mask(df)` so writers
(update_fundamentals.py, backfill_edgar.py) can reject absurd values at
ingestion time.
"""
import pandas as pd
from analytics_common import atomic_write_parquet
from pathlib import Path

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"

BAD_BIG = 100_000      # $100T in billions — 1e6x error
BAD_SMALL = 0.05       # $50M in billions — 1e3x error for large caps


def bad_marketcap_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows whose market_cap_b is an absurd unit error."""
    if "market_cap_b" not in df.columns:
        return pd.Series(False, index=df.index)
    m = df["market_cap_b"].astype(float)
    return (m > BAD_BIG) | ((m > 0) & (m < BAD_SMALL))


def main() -> None:
    df = pd.read_parquet(FUND)
    n0 = len(df)
    mask = bad_marketcap_mask(df)
    if mask.sum() == 0:
        print("No bad market_cap rows found — nothing to do.")
        return
    bad = df.loc[mask, ["ticker", "as_of_date", "market_cap", "market_cap_b", "source"]]
    print(f"Found {mask.sum()} bad rows to null:")
    print(bad.to_string(index=False))

    df.loc[mask, ["market_cap", "market_cap_b"]] = None
    atomic_write_parquet(df, FUND)

    # verify
    chk = pd.read_parquet(FUND)
    remaining = bad_marketcap_mask(chk).sum()
    print(f"\nNulled {mask.sum()} rows ({n0} -> {len(chk)}). Remaining bad: {remaining}")


if __name__ == "__main__":
    main()

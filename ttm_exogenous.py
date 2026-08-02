#!/usr/bin/env python3
"""
ttm_exogenous.py — Build exogenous feature channels for Granite TTM forecasts.

Exogenous sources (from local parquet; no network required):
  - Equal-weight market return of all monitored names
  - Sector equal-weight returns
  - Cross-sectional dispersion (cross-sec vol)
  - Broad vol proxy (avg 20d realized vol)
  - Optional CSV of external series (--from-csv date,col1,col2,...)

Usage:
  python ttm_exogenous.py --save
  python ttm_exogenous.py --ticker MOS --save
  from ttm_exogenous import build_exog_panel, merge_exog
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
EXOG_FILE = DATA_DIR / "exogenous_panel.parquet"


def _load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.sort_values(["ticker", "date"])


def build_exog_panel(
    start: Optional[pd.Timestamp] = None,
    external_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Columns (business-day indexed):
      mkt_ret      — equal-weight log return of all tickers
      mkt_vol20    — 20d realized vol of mkt_ret
      dispersion   — cross-sectional std of daily returns
      sector_<name>_ret — EW sector returns (key sectors)
      plus any columns from external_csv
    """
    prices = _load_prices()
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    if start is not None:
        wide = wide[wide.index >= pd.Timestamp(start)]
    # business day grid
    if len(wide) >= 2:
        wide = wide.reindex(pd.bdate_range(wide.index.min(), wide.index.max())).ffill()

    logret = np.log(wide / wide.shift(1))
    mkt_ret = logret.mean(axis=1)
    dispersion = logret.std(axis=1)
    mkt_vol20 = mkt_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    exog = pd.DataFrame({
        "mkt_ret": mkt_ret,
        "mkt_vol20": mkt_vol20,
        "dispersion": dispersion,
    })

    # Sector EW returns
    if STOCKS_FILE.exists():
        stocks = pd.read_parquet(STOCKS_FILE)
        if "sector" in stocks.columns:
            for sector, grp in stocks.groupby("sector"):
                cols = [t for t in grp["ticker"] if t in logret.columns]
                if len(cols) < 2:
                    continue
                key = "sector_" + "".join(ch if ch.isalnum() else "_" for ch in sector)[:20]
                exog[key] = logret[cols].mean(axis=1)

    if external_csv is not None and Path(external_csv).exists():
        ext = pd.read_csv(external_csv, parse_dates=True)
        # expect a date column
        dcol = "date" if "date" in ext.columns else ext.columns[0]
        ext[dcol] = pd.to_datetime(ext[dcol])
        ext = ext.set_index(dcol).sort_index()
        for c in ext.columns:
            exog[f"ext_{c}"] = ext[c].reindex(exog.index).ffill()

    return exog.dropna(how="all")


def merge_exog(
    panel: pd.DataFrame,
    exog: Optional[pd.DataFrame] = None,
    cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Left-join exogenous channels onto a ticker/feature panel (index=date)."""
    if exog is None:
        if EXOG_FILE.exists():
            exog = pd.read_parquet(EXOG_FILE)
            if 'date' in exog.columns:
                exog['date'] = pd.to_datetime(exog['date'])
                exog = exog.set_index('date')
            else:
                exog.index = pd.to_datetime(exog.index)
        else:
            exog = build_exog_panel()
    if cols:
        use = [c for c in cols if c in exog.columns]
        exog = exog[use]
    out = panel.join(exog, how="left")
    # ffill exog only
    for c in exog.columns:
        if c in out.columns:
            out[c] = out[c].ffill()
    return out


def main():
    parser = argparse.ArgumentParser(description="Build exogenous panel for TTM")
    parser.add_argument("--from-csv", help="Optional external CSV with date column")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    exog = build_exog_panel(external_csv=Path(args.from_csv) if args.from_csv else None)
    print(f"Exogenous panel: {exog.shape[0]} days × {exog.shape[1]} channels")
    print(f"  {exog.index.min().date()} → {exog.index.max().date()}")
    print(f"  columns: {list(exog.columns)}")
    print(exog.tail(3).round(4).to_string())
    if args.save:
        out = exog.copy(); out.index.name = 'date'
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(out.reset_index(), preserve_index=False), EXOG_FILE)
        print(f"Saved {EXOG_FILE} ({len(out)} rows)")


if __name__ == "__main__":
    main()

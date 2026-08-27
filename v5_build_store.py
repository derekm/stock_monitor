#!/usr/bin/env python3
"""
v5_build_store.py — build the V5 feature store from the repo's real tables.

WHY

v5_integrated.V5Pipeline reads everything from a ParquetFeatureStore. No store existed
on disk, so the pipeline had only ever run against the synthetic data in its own
self-test. This builds the store from real inputs:

  panel         daily_prices/  -> per (date, ticker) features
  returns_wide  daily_prices/  -> date x ticker adj_close returns
  sectors       sp500_constituents.parquet gics_sector (real GICS), plus
                monitored_stocks.parquet sp500_sector as a secondary source
  adv           daily_prices/  -> median dollar volume, trailing window
  borrow_bps    NOT AVAILABLE -> written as a constant and reported as such

FEATURES

The six V5Config defaults, computed per ticker on adjusted closes:
  ret_1, ret_5, ret_10   trailing simple returns
  vol_10, vol_20         trailing stdev of daily returns
  ma_gap                 close / 20d moving average - 1

All are backward-looking as of `date`; forward labels are built later by the ranker
from returns_wide, so the panel itself carries no future information.

HONEST GAPS, stated rather than papered over:
  - borrow_bps has no source in this repo. Every ticker gets --borrow-bps (default
    150, the config's soft threshold). Short-side borrow costs are therefore uniform,
    not real, and any short P&L is only as good as that assumption.
  - sectors cover the S&P 500 names only. Tickers with no GICS sector are dropped
    when --require-sector is set (the default), because sector-neutralization on an
    "Unknown" bucket silently neutralizes against a meaningless group.

Usage:
    python v5_build_store.py --start 2016-01-01
    python v5_build_store.py --start 2016-01-01 --min-price 5 --min-adv 5e6
    python v5_build_store.py --start 2020-01-01 --store-root ./v5_feature_store
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices/"
SP500 = DATA_DIR / "sp500_constituents.parquet"
MONITORED = DATA_DIR / "monitored_stocks.parquet"

FEATURE_COLS = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]


def load_sectors() -> pd.Series:
    """ticker -> GICS sector, from the real constituent table."""
    sec = {}
    sp = pl.read_parquet(SP500).select(["ticker", "gics_sector"]).to_pandas()
    for t, s in zip(sp["ticker"], sp["gics_sector"]):
        if isinstance(s, str) and s.strip():
            sec[t] = s.strip()
    # monitored_stocks carries sp500_sector for names outside the index snapshot
    if MONITORED.exists():
        try:
            mon = pl.read_parquet(MONITORED).to_pandas()
            col = "sp500_sector" if "sp500_sector" in mon.columns else "sector"
            for t, s in zip(mon["ticker"], mon[col]):
                if t not in sec and isinstance(s, str) and s.strip():
                    sec[t] = s.strip()
        except Exception:
            pass
    return pd.Series(sec, name="sector")


def build(start: str, min_price: float, min_adv: float, adv_window: int,
          require_sector: bool) -> tuple:
    px = (pl.read_parquet(PRICES, columns=["date", "ticker", "adj_close", "volume"])
          .filter(pl.col("date") >= pd.Timestamp(start).date())
          .filter(pl.col("adj_close").is_not_null() & (pl.col("adj_close") > 0))
          .sort(["ticker", "date"])
          .to_pandas())
    px["date"] = pd.to_datetime(px["date"])
    print(f"prices: {len(px):,} rows, {px['ticker'].nunique():,} tickers, "
          f"{px['date'].min().date()} -> {px['date'].max().date()}")

    sectors = load_sectors()
    print(f"sectors: {len(sectors):,} tickers mapped, "
          f"{sectors.nunique()} distinct GICS sectors")

    if require_sector:
        keep = set(sectors.index)
        before = px["ticker"].nunique()
        px = px[px["ticker"].isin(keep)]
        print(f"  restricted to sector-mapped tickers: {before:,} -> "
              f"{px['ticker'].nunique():,}")

    # liquidity / price screens, applied on the ticker's own median
    px["dollar_vol"] = px["adj_close"] * px["volume"].fillna(0.0)
    med = px.groupby("ticker").agg(med_px=("adj_close", "median"),
                                   med_dv=("dollar_vol", "median"))
    ok = med[(med["med_px"] >= min_price) & (med["med_dv"] >= min_adv)].index
    before = px["ticker"].nunique()
    px = px[px["ticker"].isin(ok)]
    print(f"  liquidity screen (px>={min_price}, adv>={min_adv:,.0f}): "
          f"{before:,} -> {px['ticker'].nunique():,} tickers")

    # ---- features, strictly backward-looking
    g = px.groupby("ticker", sort=False)
    px["r1"] = g["adj_close"].pct_change()
    px["ret_1"] = px["r1"]
    px["ret_5"] = g["adj_close"].pct_change(5)
    px["ret_10"] = g["adj_close"].pct_change(10)
    px["vol_10"] = g["r1"].transform(lambda s: s.rolling(10, min_periods=8).std())
    px["vol_20"] = g["r1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    ma20 = g["adj_close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    px["ma_gap"] = px["adj_close"] / ma20 - 1.0

    panel = px[["date", "ticker"] + FEATURE_COLS].dropna().reset_index(drop=True)
    print(f"panel: {len(panel):,} rows after dropping incomplete features")

    returns_wide = (px.pivot_table(index="date", columns="ticker", values="adj_close")
                    .sort_index().pct_change())
    print(f"returns_wide: {returns_wide.shape[0]:,} dates x "
          f"{returns_wide.shape[1]:,} tickers")

    # ADV: median dollar volume over the trailing window, per ticker
    recent = px[px["date"] >= px["date"].max() - pd.Timedelta(days=adv_window)]
    adv = recent.groupby("ticker")["dollar_vol"].median().rename("adv")
    adv = adv[adv > 0]

    tickers = sorted(set(panel["ticker"]) & set(adv.index))
    panel = panel[panel["ticker"].isin(tickers)]
    returns_wide = returns_wide[[t for t in returns_wide.columns if t in tickers]]
    adv = adv.loc[tickers]
    sectors_out = sectors.reindex(tickers).dropna()

    return panel, returns_wide, sectors_out, adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-adv", type=float, default=5e6)
    ap.add_argument("--adv-window", type=int, default=90,
                    help="calendar days for the ADV median")
    ap.add_argument("--borrow-bps", type=float, default=150.0,
                    help="UNIFORM assumed annual borrow cost; no real source exists")
    ap.add_argument("--store-root", default="./v5_feature_store")
    ap.add_argument("--require-sector", action="store_true", default=True)
    ap.add_argument("--allow-unknown-sector", dest="require_sector",
                    action="store_false")
    args = ap.parse_args()

    panel, returns_wide, sectors, adv = build(
        args.start, args.min_price, args.min_adv, args.adv_window,
        args.require_sector)

    borrow = pd.Series(args.borrow_bps, index=adv.index, name="borrow_bps_annual")

    print()
    print("FINAL INPUTS")
    print(f"  panel        {panel.shape}")
    print(f"  returns_wide {returns_wide.shape}")
    print(f"  sectors      {len(sectors)} mapped, {sectors.nunique()} sectors")
    print(f"  adv          {len(adv)} tickers, median "
          f"${adv.median()/1e6:,.1f}M")
    print(f"  borrow       CONSTANT {args.borrow_bps:.0f} bps -- assumption, not data")
    print()
    print("sector distribution:")
    for s, n in sectors.value_counts().items():
        print(f"    {s:26} {n:4}")

    from feature_store import create_store_from_data
    store = create_store_from_data(
        panel, returns_wide, sectors, adv, borrow,
        store_root=args.store_root, feature_cols=FEATURE_COLS,
        run_name="v5_real_data",
    )
    print()
    print(f"store written: {Path(args.store_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
allpairs_correlations.py — Dense pairwise asset & sector correlations over time.

Builds:
  - Full asset ALLPAIRS corr on trailing windows (stacked long format)
  - Sector EW ALLPAIRS over the same windows
  - Latest wide matrices for dashboard

Usage:
  python allpairs_correlations.py
  python allpairs_correlations.py --window 63 --step 21 --max-assets 80
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_ASSET = DATA_DIR / "allpairs_asset_corr_history.csv"
OUT_SECTOR = DATA_DIR / "allpairs_sector_corr_history.csv"
OUT_ASSET_LATEST = DATA_DIR / "allpairs_asset_corr_latest.csv"
OUT_SECTOR_LATEST = DATA_DIR / "allpairs_sector_corr_latest.csv"
OUT_SUMMARY = DATA_DIR / "allpairs_corr_summary.csv"


def pairwise_long(corr: pd.DataFrame, date, window: int, kind: str) -> list[dict]:
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append({
                "date": date,
                "window": window,
                "kind": kind,
                "asset_a": a,
                "asset_b": b,
                "corr": float(corr.loc[a, b]) if pd.notna(corr.loc[a, b]) else np.nan,
            })
    return rows


def run(window: int = 63, step: int = 21, max_assets: int = 80, save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS, columns=["ticker", "sector", "defensive_value_index", "growth_tech_index", "dual_pass_member"])

    # prioritize dual + portfolio-ish + defensive + growth sample
    priority = []
    for col in ["dual_pass_member", "defensive_value_index", "growth_tech_index"]:
        if col in stocks.columns:
            priority += stocks.loc[stocks[col] == True, "ticker"].tolist()
    # fill with most liquid by row count
    counts = prices.groupby("ticker").size().sort_values(ascending=False)
    ordered = list(dict.fromkeys(priority + counts.index.tolist()))
    tickers = ordered[:max_assets]

    wide = (
        prices[prices.ticker.isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    rets = np.log(wide / wide.shift(1))
    dates = rets.index[window::step]
    if len(dates) == 0:
        dates = rets.index[window - 1 :][-1:]

    asset_rows = []
    sector_rows = []
    sector_map = stocks.set_index("ticker")["sector"].to_dict()

    for dt in dates:
        loc = rets.index.get_loc(dt)
        if isinstance(loc, slice):
            continue
        block = rets.iloc[max(0, loc - window + 1) : loc + 1]
        c = block.corr()
        asset_rows.extend(pairwise_long(c, dt, window, "asset"))

        # sector EW
        sret = {}
        for sec in sorted(set(sector_map.values())):
            cols = [t for t in c.columns if sector_map.get(t) == sec]
            if len(cols) >= 2:
                sret[sec] = block[cols].mean(axis=1)
        if len(sret) >= 2:
            sc = pd.DataFrame(sret).corr()
            sector_rows.extend(pairwise_long(sc, dt, window, "sector"))

    asset_df = pd.DataFrame(asset_rows)
    sector_df = pd.DataFrame(sector_rows)
    print(f"ALLPAIRS asset pairs: {len(asset_df)} rows across {asset_df.date.nunique() if len(asset_df) else 0} dates")
    print(f"ALLPAIRS sector pairs: {len(sector_df)} rows")

    # latest wide
    if len(asset_df):
        last = asset_df.date.max()
        latest = asset_df[asset_df.date == last]
        # pivot to wide
        assets = sorted(set(latest.asset_a) | set(latest.asset_b))
        mat = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
        for _, r in latest.iterrows():
            mat.loc[r.asset_a, r.asset_b] = r['corr']
            mat.loc[r.asset_b, r.asset_a] = r['corr']
        mat.to_csv(OUT_ASSET_LATEST)
        print(f"Latest asset matrix {mat.shape} @ {last.date()}")

    if len(sector_df):
        last = sector_df.date.max()
        latest = sector_df[sector_df.date == last]
        secs = sorted(set(latest.asset_a) | set(latest.asset_b))
        mat = pd.DataFrame(np.eye(len(secs)), index=secs, columns=secs)
        for _, r in latest.iterrows():
            mat.loc[r.asset_a, r.asset_b] = r['corr']
            mat.loc[r.asset_b, r.asset_a] = r['corr']
        mat.to_csv(OUT_SECTOR_LATEST)
        print(f"Latest sector matrix {mat.shape} @ {last.date()}")

    # summary stats
    summary = []
    if len(asset_df):
        summary.append({
            "kind": "asset",
            "mean_corr": asset_df['corr'].mean(),
            "median_corr": asset_df['corr'].median(),
            "p10": asset_df['corr'].quantile(0.1),
            "p90": asset_df['corr'].quantile(0.9),
            "std": asset_df['corr'].std(),
            "n_pairs": len(asset_df),
            "n_dates": asset_df.date.nunique(),
        })
    if len(sector_df):
        summary.append({
            "kind": "sector",
            "mean_corr": sector_df['corr'].mean(),
            "median_corr": sector_df['corr'].median(),
            "p10": sector_df['corr'].quantile(0.1),
            "p90": sector_df['corr'].quantile(0.9),
            "std": sector_df['corr'].std(),
            "n_pairs": len(sector_df),
            "n_dates": sector_df.date.nunique(),
        })
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))

    if save:
        # keep history manageable
        if len(asset_df) > 200_000:
            asset_df = asset_df.tail(200_000)
        asset_df.to_csv(OUT_ASSET, index=False)
        sector_df.to_csv(OUT_SECTOR, index=False)
        sdf.to_csv(OUT_SUMMARY, index=False)
        print(f"Wrote {OUT_ASSET}, {OUT_SECTOR}, {OUT_SUMMARY}")
    return asset_df, sector_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--max-assets", type=int, default=60)
    args = ap.parse_args()
    run(window=args.window, step=args.step, max_assets=args.max_assets, save=True)


if __name__ == "__main__":
    main()

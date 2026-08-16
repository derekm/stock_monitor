#!/usr/bin/env python3
"""
allpairs_correlations.py — Dense pairwise asset & sector correlations over time.

Vectorized implementation:
  - Full asset ALLPAIRS corr on trailing windows (stacked long format)
  - Sector EW ALLPAIRS over the same windows
  - Latest wide matrices for dashboard

Performance: O(N·k²) per window step via numpy broadcasting (was O(N·k²/2) with
Python dict-appends). All correlation matrices computed in bulk.

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
OUT_ASSET = DATA_DIR / "allpairs_asset_corr_history.parquet"
OUT_SECTOR = DATA_DIR / "allpairs_sector_corr_history.parquet"
OUT_ASSET_LATEST = DATA_DIR / "allpairs_asset_corr_latest.parquet"
OUT_SECTOR_LATEST = DATA_DIR / "allpairs_sector_corr_latest.parquet"
OUT_SUMMARY = DATA_DIR / "allpairs_corr_summary.parquet"


def pairwise_long(corr: np.ndarray, cols: list[str], date, window: int, kind: str) -> list[dict]:
    """Upper-triangular pairs of a correlation matrix, vectorized.
    
    np.triu_indices gives all (i, j) with j > i in one shot.
    """
    k = len(cols)
    if k < 2:
        return []
    i_idx, j_idx = np.triu_indices(k, k=1)
    vals = corr
    return [
        {
            "date": date,
            "window": window,
            "kind": kind,
            "asset_a": cols[i],
            "asset_b": cols[j],
            "corr": float(vals[i, j]) if np.isfinite(vals[i, j]) else np.nan,
        }
        for i, j in zip(i_idx, j_idx)
    ]


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
        dates = rets.index[window - 1:][-1:]

    # Pre-compute sector mapping and column indices
    sector_map = stocks.set_index("ticker")["sector"].to_dict()
    sector_cols = {sec: [t for t in tickers if sector_map.get(t) == sec]
                   for sec in sorted(set(sector_map.values()))}
    
    # Pre-compute sector column indices for fast EW calculation
    sector_col_indices = {}
    all_cols = list(wide.columns)
    col_to_idx = {c: i for i, c in enumerate(all_cols)}
    for sec, cols in sector_cols.items():
        if len(cols) >= 2:
            sector_col_indices[sec] = [col_to_idx[c] for c in cols if c in col_to_idx]

    rets_np = rets.to_numpy()  # (N, k)
    asset_rows = []
    sector_rows = []

    for dt in dates:
        loc = rets.index.get_loc(dt)
        if isinstance(loc, slice):
            continue
        start = max(0, loc - window + 1)
        block = rets_np[start:loc + 1]  # (window, k)
        
        # Vectorized correlation: standardize then X.T @ X / (n-1)
        valid = ~np.isnan(block).any(axis=0)
        if valid.sum() < 2:
            continue
        block_v = block[:, valid]
        block_v = block_v - block_v.mean(axis=0)
        std = block_v.std(axis=0, ddof=1)
        std[std == 0] = 1
        block_v = block_v / std
        corr = block_v.T @ block_v / (block_v.shape[0] - 1)
        np.fill_diagonal(corr, 1.0)
        
        valid_cols = [c for i, c in enumerate(all_cols) if valid[i]]
        asset_rows.extend(pairwise_long(corr, valid_cols, dt, window, "asset"))

        # Sector EW — vectorized using pre-computed indices
        sret_dict = {}
        for sec, indices in sector_col_indices.items():
            # Filter to valid columns only
            sec_valid = [i for i in indices if valid[i]]
            if len(sec_valid) >= 2:
                sret_dict[sec] = block[:, sec_valid].mean(axis=1)
        
        if len(sret_dict) >= 2:
            sret_df = pd.DataFrame(sret_dict)
            sc = sret_df.corr()
            sector_rows.extend(pairwise_long(sc.values, list(sc.columns), dt, window, "sector"))

    asset_df = pd.DataFrame(asset_rows)
    sector_df = pd.DataFrame(sector_rows)
    print(f"ALLPAIRS asset pairs: {len(asset_df)} rows across {asset_df.date.nunique() if len(asset_df) else 0} dates")
    print(f"ALLPAIRS sector pairs: {len(sector_df)} rows")

    # latest wide — vectorized construction
    if len(asset_df):
        last = asset_df.date.max()
        latest = asset_df[asset_df.date == last]
        assets = sorted(set(latest.asset_a) | set(latest.asset_b))
        n = len(assets)
        mat = np.eye(n)
        idx_map = {a: i for i, a in enumerate(assets)}
        for _, r in latest.iterrows():
            i, j = idx_map[r.asset_a], idx_map[r.asset_b]
            mat[i, j] = r['corr']
            mat[j, i] = r['corr']
        mat_df = pd.DataFrame(mat, index=assets, columns=assets)
        mat_df.to_parquet(OUT_ASSET_LATEST)
        print(f"Latest asset matrix {mat_df.shape} @ {last.date()}")

    if len(sector_df):
        last = sector_df.date.max()
        latest = sector_df[sector_df.date == last]
        secs = sorted(set(latest.asset_a) | set(latest.asset_b))
        n = len(secs)
        mat = np.eye(n)
        idx_map = {s: i for i, s in enumerate(secs)}
        for _, r in latest.iterrows():
            i, j = idx_map[r.asset_a], idx_map[r.asset_b]
            mat[i, j] = r['corr']
            mat[j, i] = r['corr']
        mat_df = pd.DataFrame(mat, index=secs, columns=secs)
        mat_df.to_parquet(OUT_SECTOR_LATEST)
        print(f"Latest sector matrix {mat_df.shape} @ {last.date()}")

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
        if len(asset_df) > 200_000:
            asset_df = asset_df.tail(200_000)
        asset_df.to_parquet(OUT_ASSET, index=False)
        sector_df.to_parquet(OUT_SECTOR, index=False)
        sdf.to_parquet(OUT_SUMMARY, index=False)
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
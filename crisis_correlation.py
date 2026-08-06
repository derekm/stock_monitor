#!/usr/bin/env python3
"""
crisis_correlation.py — Correlation breakdown in stress / crisis regimes.
Optimized with Polars for data loading, numpy for correlation computation.

Defines crisis windows as:
  1) Top-quintile market vol days (realized 21d vol)
  2) Worst 5% market return days
  3) Drawdown episodes (market below -8% from peak)

Compares avg pairwise corr: calm vs crisis.

Usage:
  python crisis_correlation.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT = DATA_DIR / "crisis_correlation_summary.csv"
OUT_PAIR = DATA_DIR / "crisis_correlation_pairs.csv"
OUT_TS = DATA_DIR / "crisis_avg_corr_timeseries.csv"


def avg_pairwise_pl(corr: np.ndarray) -> float:
    """Average of upper triangle of correlation matrix."""
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    return float(np.nanmean(corr[mask]))


def run(save: bool = True):
    # Load data with Polars
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close"])
    stocks = pl.read_parquet(STOCKS, columns=["ticker", "sector", "defensive_value_index", "growth_tech_index", "value_sleeve", "instrument_type"])
    
    # Pivot to wide format (date x ticker) using Polars
    wide = prices.pivot(index="date", on="ticker", values="close").sort("date")
    
    # Forward fill then compute log returns using numpy
    wide_ff = wide.fill_null(strategy="forward")
    tickers = [c for c in wide_ff.columns if c != "date"]
    
    # Convert to numpy for fast correlation computation
    dates = wide_ff["date"].to_numpy()
    price_matrix = wide_ff.select(pl.exclude("date")).to_numpy()
    
    # Compute log returns
    log_rets = np.log(price_matrix[1:] / price_matrix[:-1])
    rets_dates = dates[1:]
    
    # Market return (equal weight)
    mkt = np.nanmean(log_rets, axis=1)
    
    # 21d rolling vol
    mkt_series = pl.Series(mkt)
    vol21 = mkt_series.rolling_std(window_size=21) * np.sqrt(252)
    vol21 = vol21.to_numpy()
    
    # Regime masks
    vol_cut = np.nanquantile(vol21, 0.8)
    crisis_vol = vol21 >= vol_cut
    crisis_ret = mkt <= np.nanquantile(mkt, 0.05)
    
    # Drawdown
    cumsum = np.cumsum(mkt)
    peak = np.maximum.accumulate(cumsum)
    dd = cumsum - peak
    crisis_dd = dd <= np.nanquantile(dd, 0.1)
    
    crisis = crisis_vol | crisis_ret | crisis_dd
    calm = ~crisis & ~np.isnan(vol21)
    
    # Returns array for correlation
    rets_arr = log_rets
    ticker_list = tickers
    
    def corr_in(mask: np.ndarray):
        idx = np.where(mask)[0]
        if len(idx) < 15:
            return None, None, None, 0
        block = rets_arr[idx, :]
        # Remove columns with all NaN
        valid_cols = ~np.all(np.isnan(block), axis=0)
        if valid_cols.sum() < 2:
            return None, None, None, 0
        block = block[:, valid_cols]
        # Compute correlation
        c = np.corrcoef(block, rowvar=False)
        valid_tickers = [t for i, t in enumerate(ticker_list) if valid_cols[i]]
        return c, valid_tickers, valid_cols, len(idx)
    
    results = []
    pair_rows = []
    
    for name, mask in [("calm", calm), ("crisis_vol", crisis_vol), ("crisis_ret", crisis_ret),
                       ("crisis_dd", crisis_dd), ("crisis_any", crisis)]:
        c, valid_tickers, valid_cols, n = corr_in(mask)
        if c is None:
            results.append({"regime": name, "n_days": n, "avg_pairwise_corr": None})
            continue
        avg = avg_pairwise_pl(c)
        # median of upper triangle
        n_tickers = c.shape[0]
        mask_tri = np.triu(np.ones((n_tickers, n_tickers), dtype=bool), 1)
        median_val = float(np.nanmedian(c[mask_tri]))
        results.append({"regime": name, "n_days": n, "avg_pairwise_corr": avg, "median_pairwise": median_val})
        print(f"{name:12s} days={n:4d}  avg_corr={avg:.3f}")
    
    # Sector-level crisis vs calm
    sector_map = {row["ticker"]: row["sector"] for row in stocks.iter_rows(named=True) if row["sector"] is not None}
    
    for regime_name, mask in [("calm", calm), ("crisis_any", crisis)]:
        c, _, valid_cols, n = corr_in(mask)
        if c is None or n < 15:
            continue
        # Group by sector
        valid_tickers = [t for t in ticker_list if t in sector_map]
        sectors = sorted(set(sector_map[t] for t in valid_tickers))
        sector_rets = {}
        for sec in sectors:
            cols = [t for t in valid_tickers if sector_map.get(t) == sec]
            if len(cols) >= 2:
                col_idx = [ticker_list.index(c) for c in cols]
                sector_rets[sec] = rets_arr[:, col_idx].mean(axis=1)
        if len(sector_rets) < 2:
            continue
        sector_block = np.column_stack(list(sector_rets.values()))
        sc = np.corrcoef(sector_block, rowvar=False)
        avg = avg_pairwise_pl(sc)
        results.append({"regime": f"sector_{regime_name}", "n_days": n, "avg_pairwise_corr": avg if not np.isnan(avg) else None})
        print(f"sector_{regime_name:8s} avg_corr={avg:.3f}")
    
    # Pair-level: largest correlation increase crisis vs calm
    c_calm, calm_tickers, calm_valid_cols, n_calm = corr_in(calm)
    c_cris, cris_tickers, cris_valid_cols, n_cris = corr_in(crisis)
    if c_calm is not None and c_cris is not None:
        # Need to align the columns since they may have different valid columns
        # Find tickers that are valid in BOTH calm and crisis
        common_tickers = set(calm_tickers) & set(cris_tickers)
        if len(common_tickers) >= 2:
            # Get indices in each correlation matrix
            calm_idx = {t: i for i, t in enumerate(calm_tickers)}
            cris_idx = {t: i for i, t in enumerate(cris_tickers)}
            common_list = sorted(common_tickers)
            calm_indices = [calm_idx[t] for t in common_list]
            cris_indices = [cris_idx[t] for t in common_list]
            
            c_calm_v = c_calm[np.ix_(calm_indices, calm_indices)]
            c_cris_v = c_cris[np.ix_(cris_indices, cris_indices)]
            valid_tickers = common_list
            pair_rows = []
            n_valid = len(valid_tickers)
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    delta = float(c_cris_v[i, j] - c_calm_v[i, j])
                    pair_rows.append({
                        "asset_a": valid_tickers[i],
                        "asset_b": valid_tickers[j],
                        "corr_calm": float(c_calm_v[i, j]) if not np.isnan(c_calm_v[i, j]) else None,
                        "corr_crisis": float(c_cris_v[i, j]) if not np.isnan(c_cris_v[i, j]) else None,
                        "delta": delta if not np.isnan(delta) else None,
                    })
            # Use pandas for reliable DataFrame creation, then convert to Polars
            pairs_pd = pd.DataFrame(pair_rows)
            pairs = pl.from_pandas(pairs_pd).sort("delta", descending=True)
            print("\nLargest corr increases in crisis:")
            print(pairs.head(10))
            print("\nLargest corr decreases (diversifiers in stress):")
            print(pairs.tail(10).sort("delta"))
            if save:
                pairs.write_csv(OUT_PAIR)
    
    # Rolling avg pairwise timeseries tagged with crisis flag
    window = 21
    # Use a subset of tickers for rolling correlation (max 50 for performance)
    max_rolling_tickers = 50
    rolling_tickers = ticker_list[:max_rolling_tickers]
    rolling_indices = [ticker_list.index(t) for t in rolling_tickers]
    rolling_rets = rets_arr[:, rolling_indices]
    
    dates_list = []
    avg_corr_list = []
    mkt_vol_list = []
    crisis_list = []
    for i in range(window, len(rolling_rets)):
        block = rolling_rets[i-window:i]
        valid_cols = ~np.all(np.isnan(block), axis=0)
        if valid_cols.sum() >= 2:
            block_v = block[:, valid_cols]
            c = np.corrcoef(block_v, rowvar=False)
            avg = avg_pairwise_pl(c)
        else:
            avg = np.nan
        dt = rets_dates[i]
        dates_list.append(dt)
        avg_corr_list.append(avg if not np.isnan(avg) else np.nan)
        mkt_vol_list.append(float(vol21[i]) if i < len(vol21) and not np.isnan(vol21[i]) else np.nan)
        crisis_list.append(bool(crisis[i]) if i < len(crisis) else False)
    
    # Build DataFrame with pandas first, then convert to Polars
    ts_df = pd.DataFrame({
        "date": dates_list,
        "avg_pairwise_corr": avg_corr_list,
        "mkt_vol21": mkt_vol_list,
        "crisis": crisis_list,
    })
    # Convert to Polars
    ts = pl.from_pandas(ts_df)
    
    summary = pl.DataFrame(results)
    if save:
        summary.write_csv(OUT)
        ts.write_csv(OUT_TS)
        print(f"\nWrote {OUT}\nWrote {OUT_TS}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()
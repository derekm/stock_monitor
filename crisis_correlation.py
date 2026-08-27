#!/usr/bin/env python3
"""
crisis_correlation.py — Correlation breakdown in stress / crisis regimes.
Optimized with numpy broadcasting for correlation computation.

Vectorized implementation:
  - Sector EW proxy + numpy broadcast for pairwise correlation
  - Single-pass regime mask computation
  - Vectorized upper-triangle extraction

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
PRICES = DATA_DIR / "daily_prices/"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT = DATA_DIR / "crisis_correlation_summary.parquet"
OUT_PAIR = DATA_DIR / "crisis_correlation_pairs.parquet"
OUT_TS = DATA_DIR / "crisis_avg_corr_timeseries.parquet"


def avg_pairwise_np(corr: np.ndarray) -> float:
    """Average of upper triangle of correlation matrix (vectorized)."""
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    return float(np.nanmean(corr[mask]))


def vectorized_corr(block: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from returns block using numpy broadcasting.
    
    block: (n_obs, n_assets) array of returns
    Returns: (n_assets, n_assets) correlation matrix
    """
    # Remove columns with all NaN
    valid_cols = ~np.all(np.isnan(block), axis=0)
    if valid_cols.sum() < 2:
        return np.array([])
    block = block[:, valid_cols]
    
    # Standardize
    block = block - np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0, ddof=1)
    std[std == 0] = 1
    block = block / std
    
    # Correlation via matrix multiplication
    n = block.shape[0]
    # Handle NaN by treating as 0 (already centered)
    block_clean = np.nan_to_num(block, nan=0.0)
    corr = block_clean.T @ block_clean / (n - 1)
    np.fill_diagonal(corr, 1.0)
    return corr


def run(save: bool = True):
    # Load data with Polars
    prices = pl.read_parquet(PRICES, columns=["date", "ticker", "close"])
    from analytics_common import load_membership
    stocks = pl.from_pandas(load_membership())
    
    # Pivot to wide format (date x ticker) using Polars
    wide = prices.pivot(index="date", on="ticker", values="close").sort("date")
    
    # Forward fill then compute log returns using numpy
    wide_ff = wide.fill_null(strategy="forward")
    tickers = [c for c in wide_ff.columns if c != "date"]
    
    # Convert to numpy for fast correlation computation
    dates = pd.to_datetime(wide_ff["date"].to_numpy()).to_numpy()
    price_matrix = wide_ff.select(pl.exclude("date")).to_numpy()
    
    # Compute log returns
    log_rets = np.log(price_matrix[1:] / price_matrix[:-1])
    rets_dates = dates[1:]
    
    # Market return (equal weight)
    mkt = np.nanmean(log_rets, axis=1)
    
    # 21d rolling vol (vectorized with cumsum)
    n = len(mkt)
    x = np.nan_to_num(mkt, nan=0.0)
    valid = ~np.isnan(mkt)
    s1 = np.concatenate([[0], np.cumsum(x)])
    s2 = np.concatenate([[0], np.cumsum(x * x)])
    sc = np.concatenate([[0], np.cumsum(valid.astype(float))])
    w = 21
    sw1 = s1[w:] - s1[:-w]
    sw2 = s2[w:] - s2[:-w]
    swc = sc[w:] - sc[:-w]
    mean_w = sw1 / w
    var_w = np.maximum((sw2 - sw1 * sw1 / w) / (w - 1), 0.0)
    vol21 = np.sqrt(var_w) * np.sqrt(252)
    vol21[swc < w] = np.nan
    # Align to mkt length. sw1=sw2=swc were sliced [w:] so they already have length
    # n+1-w (one shorter than mkt due to the trailing window); pad with w-1 NaNs,
    # not w, so crisis_vol matches crisis_ret/crisis_dd (both length n). A w-pad
    # made vol21 length n+1 and broke `crisis_vol | crisis_ret | crisis_dd`.
    vol21 = np.concatenate([np.full(w - 1, np.nan), vol21])
    
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
            return None, None, 0
        block = rets_arr[idx, :]
        c = vectorized_corr(block)
        if c.size == 0:
            return None, None, 0
        valid_cols = ~np.all(np.isnan(block), axis=0)
        valid_tickers = [t for i, t in enumerate(ticker_list) if valid_cols[i]]
        return c, valid_tickers, len(idx)
    
    results = []
    pair_rows = []
    
    for name, mask in [("calm", calm), ("crisis_vol", crisis_vol), ("crisis_ret", crisis_ret),
                       ("crisis_dd", crisis_dd), ("crisis_any", crisis)]:
        c, valid_tickers, n = corr_in(mask)
        if c is None:
            results.append({"regime": name, "n_days": n, "avg_pairwise_corr": None})
            continue
        avg = avg_pairwise_np(c)
        n_tickers = c.shape[0]
        mask_tri = np.triu(np.ones((n_tickers, n_tickers), dtype=bool), 1)
        median_val = float(np.nanmedian(c[mask_tri]))
        results.append({"regime": name, "n_days": n, "avg_pairwise_corr": avg, "median_pairwise": median_val})
        print(f"{name:12s} days={n:4d}  avg_corr={avg:.3f}")
    
    # Sector-level crisis vs calm (vectorized)
    sector_map = {row["ticker"]: row["sector"] for row in stocks.iter_rows(named=True) if row["sector"] is not None}
    
    for regime_name, mask in [("calm", calm), ("crisis_any", crisis)]:
        c, valid_tickers, n = corr_in(mask)
        if c is None or n < 15:
            continue
        # Group by sector using numpy. Exclude unmapped (sector=NaN) tickers:
        # sorting a mix of float('nan') and str raises TypeError now that the
        # universe includes 6,729 names with no sector.
        valid_tickers_sec = [t for t in ticker_list
                             if isinstance(sector_map.get(t), str) and sector_map[t].strip()]
        sectors = sorted(set(sector_map[t] for t in valid_tickers_sec))
        sector_rets = {}
        for sec in sectors:
            cols = [t for t in valid_tickers_sec if sector_map.get(t) == sec]
            if len(cols) >= 2:
                col_idx = [ticker_list.index(c) for c in cols]
                sector_rets[sec] = rets_arr[:, col_idx].mean(axis=1)
        if len(sector_rets) < 2:
            continue
        sector_block = np.column_stack(list(sector_rets.values()))
        sc = vectorized_corr(sector_block[mask, :])
        if sc.size == 0:
            continue
        avg = avg_pairwise_np(sc)
        results.append({"regime": f"sector_{regime_name}", "n_days": n, "avg_pairwise_corr": avg if not np.isnan(avg) else None})
        print(f"sector_{regime_name:8s} avg_corr={avg:.3f}")
    
    # Pair-level: largest correlation increase crisis vs calm
    c_calm, calm_tickers, n_calm = corr_in(calm)
    c_cris, cris_tickers, n_cris = corr_in(crisis)
    if c_calm is not None and c_cris is not None:
        common_tickers = set(calm_tickers) & set(cris_tickers)
        if len(common_tickers) >= 2:
            calm_idx = {t: i for i, t in enumerate(calm_tickers)}
            cris_idx = {t: i for i, t in enumerate(cris_tickers)}
            common_list = sorted(common_tickers)
            calm_indices = [calm_idx[t] for t in common_list]
            cris_indices = [cris_idx[t] for t in common_list]
            
            c_calm_v = c_calm[np.ix_(calm_indices, calm_indices)]
            c_cris_v = c_cris[np.ix_(cris_indices, cris_indices)]
            
            # Vectorized pair extraction
            n_valid = len(common_list)
            i_idx, j_idx = np.triu_indices(n_valid, k=1)
            deltas = c_cris_v[i_idx, j_idx] - c_calm_v[i_idx, j_idx]
            calm_corrs = c_calm_v[i_idx, j_idx]
            cris_corrs = c_cris_v[i_idx, j_idx]
            
            pair_rows = [
                {
                    "asset_a": common_list[i],
                    "asset_b": common_list[j],
                    "corr_calm": float(calm_corrs[k]) if not np.isnan(calm_corrs[k]) else None,
                    "corr_crisis": float(cris_corrs[k]) if not np.isnan(cris_corrs[k]) else None,
                    "delta": float(deltas[k]) if not np.isnan(deltas[k]) else None,
                }
                for k, (i, j) in enumerate(zip(i_idx, j_idx))
            ]
            pairs = pl.DataFrame(pair_rows).sort("delta", descending=True)
            print("\nLargest corr increases in crisis:")
            print(pairs.head(10))
            print("\nLargest corr decreases (diversifiers in stress):")
            print(pairs.tail(10).sort("delta"))
            if save:
                pairs.write_csv(OUT_PAIR)
    
    # Rolling avg pairwise timeseries tagged with crisis flag
    window = 21
    max_rolling_tickers = 50
    rolling_tickers = ticker_list[:max_rolling_tickers]
    rolling_indices = [ticker_list.index(t) for t in rolling_tickers]
    rolling_rets = rets_arr[:, rolling_indices]
    
    # Vectorized rolling correlation using cumsum
    n_roll = len(rolling_rets)
    X = np.nan_to_num(rolling_rets, nan=0.0)
    valid = ~np.isnan(rolling_rets)
    
    s1 = np.vstack([np.zeros((1, max_rolling_tickers)), np.cumsum(X, axis=0)])
    s2 = np.vstack([np.zeros((1, max_rolling_tickers)), np.cumsum(X * X, axis=0)])
    sc = np.vstack([np.zeros((1, max_rolling_tickers)), np.cumsum(valid.astype(float), axis=0)])
    
    dates_list = []
    avg_corr_list = []
    mkt_vol_list = []
    crisis_list = []
    
    for i in range(window, n_roll):
        blk = rolling_rets[i-window:i]
        c = vectorized_corr(blk)
        avg = avg_pairwise_np(c) if c.size > 0 else np.nan
        dt = rets_dates[i]
        dates_list.append(dt)
        avg_corr_list.append(avg if not np.isnan(avg) else np.nan)
        mkt_vol_list.append(float(vol21[i]) if i < len(vol21) and not np.isnan(vol21[i]) else np.nan)
        crisis_list.append(bool(crisis[i]) if i < len(crisis) else False)
    
    ts = pl.DataFrame({
        "date": [str(d) for d in dates_list],
        "avg_pairwise_corr": avg_corr_list,
        "mkt_vol21": mkt_vol_list,
        "crisis": crisis_list,
    })
    
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
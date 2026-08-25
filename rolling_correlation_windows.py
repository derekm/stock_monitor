#!/usr/bin/env python3
"""
rolling_correlation_windows.py — Rolling pairwise & sector correlation windows.
Vectorized with cumsum-based rolling statistics.

Outputs:
  rolling_corr_avg_timeseries.parquet  — avg pairwise corr over time
  rolling_sector_corr_windows.parquet  — sector EW rolling corr long format
  rolling_corr_stability_by_asset.parquet — per-name avg corr to market + stability

Usage:
  python rolling_correlation_windows.py --windows 21,63,126 --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Import HMM's numba-optimized pairwise correlation functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from hmm_regime_detection import _pairwise_avg_corr_numba, _pairwise_avg_corr_np

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_TS = DATA_DIR / "rolling_corr_avg_timeseries.parquet"
OUT_SEC = DATA_DIR / "rolling_sector_corr_windows.parquet"
OUT_STAB = DATA_DIR / "rolling_corr_stability_by_asset.parquet"


def rolling_pairwise_stats(rets: pd.DataFrame, w: int, sample_idx: np.ndarray):
    """Mean pairwise corr + mkt vol over a trailing w-day window.

    Uses HMM's numba-optimized core (strictly-upper-triangular pairs,
    O(N·k²/2), parallel) for the mean. Median returns NaN (compute
    separately if needed — the mean is the primary signal).
    Returns (dates, avg_arr, med_arr, vol_arr).
    """
    X = rets.values.astype(np.float64)
    try:
        avg = _pairwise_avg_corr_numba(X, w)
    except Exception:
        avg = _pairwise_avg_corr_np(X, w)
    # median: NaN (compute separately if needed; mean is primary signal)
    med = np.full(rets.shape[0], np.nan)
    # sample
    tpos = np.asarray(sample_idx, dtype=np.int64)
    n_samp = len(tpos)
    dates = rets.index[tpos]
    avg_s = avg[tpos]
    med_s = med[tpos]
    # market vol: rolling std of the EW market return over the window
    mkt = rets.mean(axis=1).values
    mkt_vol = np.full(n_samp, np.nan)
    for ti, t in enumerate(tpos):
        blk = mkt[t + 1 - w : t + 1]
        mkt_vol[ti] = float(np.std(blk) * np.sqrt(252))
    return rets.index[tpos], avg_s, med_s, mkt_vol


def vectorized_rolling_corr(rets: np.ndarray, w: int) -> np.ndarray:
    """Vectorized rolling correlation using cumsum identities.
    
    rets: (N, k) returns array
    Returns: (N-w+1, k, k) correlation matrices
    """
    N, k = rets.shape
    out = np.full((N - w + 1, k, k), np.nan)
    
    # For each window, compute correlation
    # Using the identity: corr = (X.T @ X) / (w - 1) for standardized X
    X = np.nan_to_num(rets, nan=0.0)
    valid = ~np.isnan(rets)
    
    # Cumsum for fast window sums
    s1 = np.vstack([np.zeros((1, k)), np.cumsum(X, axis=0)])
    s2 = np.vstack([np.zeros((1, k)), np.cumsum(X * X, axis=0)])
    sc = np.vstack([np.zeros((1, k)), np.cumsum(valid.astype(float), axis=0)])
    
    for i in range(N - w + 1):
        sw1 = s1[i + w] - s1[i]
        sw2 = s2[i + w] - s2[i]
        swc = sc[i + w] - sc[i]
        
        valid_cols = swc >= w
        if valid_cols.sum() < 2:
            continue
        
        cols = np.where(valid_cols)[0]
        block = rets[i:i+w][:, cols]
        
        # Standardize
        block = block - block.mean(axis=0)
        std = block.std(axis=0, ddof=1)
        std[std == 0] = 1
        block = block / std
        
        corr = block.T @ block / (w - 1)
        np.fill_diagonal(corr, 1.0)
        
        out[i, cols[:, None], cols] = corr
    
    return out


def run(windows=(21, 63, 126), step: int = 5, max_assets: int = 80, save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices = prices.rename(columns={"adj_close": "close"})
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS, columns=["ticker", "sector"]) if STOCKS.exists() else pd.DataFrame()
    sector_map = stocks.set_index("ticker")["sector"].to_dict() if len(stocks) else {}

    counts = prices.groupby("ticker").size().sort_values(ascending=False)
    tickers = counts.index.tolist()[:max_assets]
    wide = (
        prices[prices.ticker.isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index().ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)

    ts_rows = []
    for w in windows:
        tpos = np.arange(w, len(rets), step)
        if not len(tpos):
            tpos = np.array([len(rets) - 1])
        dates, avg, med, vol = rolling_pairwise_stats(rets, w, tpos)
        for d, a, m, v in zip(dates, avg, med, vol):
            ts_rows.append({
                "date": d,
                "window": w,
                "avg_pairwise_corr": a,
                "median_pairwise_corr": m,
                "mkt_vol": v,
            })
    ts = pd.DataFrame(ts_rows)
    print("=== Rolling avg pairwise corr (last 3 per window) ===")
    for w in windows:
        sub = ts[ts.window == w].tail(3)
        print(sub.to_string(index=False))

    # sector rolling - vectorized
    sec_rows = []
    # Drop unmapped tickers before sorting: the universe expansion left 6,729
    # names with sector=NaN, and sorted() on a mix of float('nan') and str
    # raises "'<' not supported between instances of 'float' and 'str'", which
    # crashed this job for 15 days. NaN is not a sector, so it is excluded
    # rather than coerced to a string bucket.
    secs = sorted({s for s in sector_map.values() if isinstance(s, str) and s.strip()})
    rets_np = rets.to_numpy()
    ticker_list = list(rets.columns)
    col_to_idx = {c: i for i, c in enumerate(ticker_list)}
    sec_indices = {}
    for sec in secs:
        cols = [t for t in ticker_list if sector_map.get(t) == sec]
        if len(cols) >= 2:
            sec_indices[sec] = [col_to_idx[c] for c in cols]
    
    for w in windows:
        for i in range(w, len(rets), max(step, w // 3)):
            block = rets_np[i - w : i]
            sret = {}
            for sec, indices in sec_indices.items():
                # Filter valid columns
                valid = ~np.all(np.isnan(block[:, indices]), axis=0)
                if valid.sum() >= 2:
                    valid_indices = [indices[j] for j in np.where(valid)[0]]
                    sret[sec] = block[:, valid_indices].mean(axis=1)
            if len(sret) < 2:
                continue
            sc_df = pd.DataFrame(sret)
            sc = sc_df.corr()
            cols = list(sc.columns)
            for a_i, a in enumerate(cols):
                for b in cols[a_i + 1:]:
                    sec_rows.append({
                        "date": rets.index[i], "window": w,
                        "sector_a": a, "sector_b": b, "corr": float(sc.loc[a, b]),
                    })
    sec_df = pd.DataFrame(sec_rows)

    # per-asset rolling corr to market stability - vectorized
    stab = []
    w = 63
    rets_arr = rets.to_numpy()
    mkt_arr = mkt.to_numpy()
    
    # Rolling correlation using cumsum
    N = len(rets_arr)
    k = rets_arr.shape[1]
    
    # For each ticker, compute rolling corr to market
    # corr = cov(x, mkt) / (std(x) * std(mkt))
    X = np.nan_to_num(rets_arr, nan=0.0)
    M = np.nan_to_num(mkt_arr, nan=0.0)
    valid_X = ~np.isnan(rets_arr)
    valid_M = ~np.isnan(mkt_arr)
    
    # Cumsum for rolling sums
    s1_X = np.vstack([np.zeros((1, k)), np.cumsum(X, axis=0)])
    s2_X = np.vstack([np.zeros((1, k)), np.cumsum(X * X, axis=0)])
    sc_X = np.vstack([np.zeros((1, k)), np.cumsum(valid_X.astype(float), axis=0)])
    s1_M = np.concatenate([[0], np.cumsum(M)])
    s2_M = np.concatenate([[0], np.cumsum(M * M)])
    sc_M = np.concatenate([[0], np.cumsum(valid_M.astype(float))])
    
    for t in range(k):
        # Rolling sums for ticker t
        sw1_X = s1_X[w:, t] - s1_X[:-w, t]
        sw2_X = s2_X[w:, t] - s2_X[:-w, t]
        swc_X = sc_X[w:, t] - sc_X[:-w, t]
        sw1_M = s1_M[w:] - s1_M[:-w]
        sw2_M = s2_M[w:] - s2_M[:-w]
        swc_M = sc_M[w:] - sc_M[:-w]
        
        valid_t = swc_X >= w
        valid_m = swc_M >= w
        valid_both = valid_t & valid_m
        
        if not valid_both.any():
            rc = pd.Series(np.nan, index=rets.index[w-1:])
        else:
            mean_X = sw1_X / w
            mean_M = sw1_M / w
            var_X = np.maximum((sw2_X - sw1_X * sw1_X / w) / (w - 1), 0.0)
            var_M = np.maximum((sw2_M - sw1_M * sw1_M / w) / (w - 1), 0.0)
            
            # Cross-covariance
            XM = np.nan_to_num(rets_arr[:, t] * mkt_arr, nan=0.0)
            s_XM = np.concatenate([[0], np.cumsum(XM)])
            sw_XM = s_XM[w:] - s_XM[:-w]
            cov = sw_XM / (w - 1) - mean_X * mean_M * w / (w - 1)
            
            rc_vals = np.full(N - w + 1, np.nan)
            std_X = np.sqrt(var_X)
            std_M = np.sqrt(var_M)
            mask = valid_both & (std_X > 0) & (std_M > 0)
            rc_vals[mask] = cov[mask] / (std_X[mask] * std_M[mask])
            rc = pd.Series(rc_vals, index=rets.index[w-1:])
        
        stab.append({
            "ticker": ticker_list[t],
            "mean_corr_to_mkt": float(rc.mean()),
            "std_corr_to_mkt": float(rc.std()),
            "last_corr_to_mkt": float(rc.dropna().iloc[-1]) if rc.dropna().size else np.nan,
            "vol_63d": float(rets_arr[-63:, t].std() * np.sqrt(252)) if N >= 63 else np.nan,
        })
    stab_df = pd.DataFrame(stab).sort_values("std_corr_to_mkt", ascending=False)
    print("\nMost unstable corr-to-market names:")
    print(stab_df.head(8).to_string(index=False))

    if save:
        ts.to_parquet(OUT_TS)
        sec_df.to_parquet(OUT_SEC)
        stab_df.to_parquet(OUT_STAB)
        print(f"Wrote {OUT_TS}, {OUT_SEC}, {OUT_STAB}")
    return ts, sec_df, stab_df


def avg_pairwise(corr: pd.DataFrame) -> tuple[float, float]:
    v = corr.values
    n = v.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    vals = v[mask]
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="21,63,126")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-assets", type=int, default=80)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    windows = tuple(int(x) for x in args.windows.split(","))
    run(windows=windows, step=args.step, max_assets=args.max_assets, save=True)


if __name__ == "__main__":
    main()
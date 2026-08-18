#!/usr/bin/env python3
"""
rolling_window_analysis.py — Rolling vol, beta, Sharpe, max-DD, dual-screen stability.

Vectorized implementation using cumsum-based rolling on wide [dates x tickers] matrix.
"""
from __future__ import annotations
import argparse
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
OUT = DATA_DIR / "rolling_window_metrics.parquet"
OUT_STAB = DATA_DIR / "rolling_screen_stability.parquet"


def resolve(universe: str) -> list[str]:
    from analytics_common import load_membership
    stocks = load_membership()
    if universe == "all":
        prices = pd.read_parquet(PRICES, columns=["ticker"])
        return prices["ticker"].unique().tolist()
    if universe == "portfolio" and HOLDINGS.exists():
        return pd.read_parquet(HOLDINGS)["ticker"].tolist()
    if universe in ("growth", "growth_tech"):
        return stocks.loc[stocks.get("growth_tech_index", False) == True, "ticker"].tolist()
    if universe == "defensive":
        return stocks.loc[stocks.get("defensive_value_index", False) == True, "ticker"].tolist()
    if universe == "aerospace":
        mask = stocks["sector"].isin(["Industrials", "Information Technology"]) & (
            stocks["industry"].astype(str).str.contains("Aerospace|Defense|Semiconductor|Electronic", case=False, na=False)
            | stocks.get("growth_sleeve", pd.Series(dtype=object)).isin(["launch_services", "starlink_supply", "maritime_launch"])
        )
        return stocks.loc[mask, "ticker"].tolist()
    return stocks["ticker"].tolist()


def rolling_cumsum_2d(arr: np.ndarray, window: int, device=None) -> np.ndarray:
    """Rolling sum for 2D array [dates, tickers] using GPU when beneficial.

    Args:
        arr: [dates, tickers] array
        window: rolling window size
        device: torch.device (cuda, directml, cpu) or None for auto
    """
    n_dates, n_tickers = arr.shape
    if n_dates < window:
        return np.full_like(arr, np.nan)
    # tensor_ops owns device selection AND the CPU path, so there is no local
    # fallback to maintain here. It sums over the LAST axis, so transpose the
    # [dates, tickers] panel into [tickers, dates] and back.
    if n_tickers * n_dates > 200_000:
        from tensor_ops import rolling_sum
        out = rolling_sum(np.nan_to_num(arr).T, window, device=device)
        return np.asarray(out).T
    cumsum = np.nancumsum(arr, axis=0)
    result = np.full_like(arr, np.nan)
    result[window-1:] = cumsum[window-1:] - np.vstack([np.zeros((1, n_tickers)), cumsum[:-window]])
    return result


def rolling_mean_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean for 2D array."""
    return rolling_cumsum_2d(arr, window) / window


def rolling_std_2d(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std for 2D array using cumsum of values and squared values."""
    n_dates, n_tickers = arr.shape
    if n_dates < window:
        return np.full_like(arr, np.nan)
    cumsum = np.nancumsum(arr, axis=0)
    sum_w = np.full_like(arr, np.nan)
    sum_w[window-1:] = cumsum[window-1:] - np.vstack([np.zeros((1, n_tickers)), cumsum[:-window]])
    cumsum_sq = np.nancumsum(arr * arr, axis=0)
    sum_sq_w = np.full_like(arr, np.nan)
    sum_sq_w[window-1:] = cumsum_sq[window-1:] - np.vstack([np.zeros((1, n_tickers)), cumsum_sq[:-window]])
    mean = sum_w / window
    var = (sum_sq_w / window) - (mean * mean)
    return np.sqrt(np.maximum(var, 0))


def rolling_beta_2d(ret: np.ndarray, mkt_ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling beta for each ticker vs market return."""
    n_dates, n_tickers = ret.shape
    if n_dates < window:
        return np.full_like(ret, np.nan)
    mkt = mkt_ret.reshape(-1, 1) if mkt_ret.ndim == 1 else mkt_ret
    mean_ret = rolling_mean_2d(ret, window)
    mean_mkt = rolling_mean_2d(mkt, window)
    d_ret = ret - mean_ret
    d_mkt = mkt - mean_mkt
    cumsum_cov = np.nancumsum(d_ret * d_mkt, axis=0)
    cov = np.full_like(ret, np.nan)
    cov[window-1:] = (cumsum_cov[window-1:] - np.vstack([np.zeros((1, n_tickers)), cumsum_cov[:-window]])) / window
    mkt_var_1d = rolling_std_2d(mkt, window) ** 2
    mkt_var = np.broadcast_to(mkt_var_1d, (n_dates, n_tickers))
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(mkt_var > 1e-12, cov / mkt_var, np.nan)
    return beta


def rolling_max_dd_2d(cum_ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling max drawdown from cumulative returns."""
    n_dates, n_tickers = cum_ret.shape
    if n_dates < window:
        return np.full_like(cum_ret, np.nan)
    result = np.full_like(cum_ret, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(cum_ret, window_shape=window, axis=0)
    peak = np.nanmax(windows, axis=2)
    current = windows[:, :, -1]
    with np.errstate(divide='ignore', invalid='ignore'):
        dd = (current - peak) / np.where(peak != 0, peak, np.nan)
    result[window-1:] = dd
    return result


def run(universe: str = "all", window: int = 63, save: bool = True, checkpoint=None):
    tickers = resolve(universe)
    if checkpoint is not None and checkpoint.is_valid(tickers) and OUT.exists():
        done = checkpoint.get_completed_tickers()
        if set(tickers).issubset(done):
            prev = pd.read_parquet(OUT)
            if "window" in prev.columns and (prev["window"] == window).all() and set(prev["ticker"]).issuperset(tickers):
                print(f"=== Rolling {window}d SKIP {len(tickers)} tickers already complete ===")
                return prev
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices = prices.rename(columns={"adj_close": "close"})
    prices["date"] = pd.to_datetime(prices["date"])
    tickers = resolve(universe)
    wide = (prices[prices.ticker.isin(tickers)]
            .pivot_table(index="date", columns="ticker", values="close")
            .sort_index().ffill())
    rets = np.log(wide / wide.shift(1))

    # Filter to tickers with sufficient data
    valid_tickers = [c for c in wide.columns if wide[c].notna().sum() >= 252]
    wide = wide[valid_tickers]
    rets = np.log(wide / wide.shift(1))

    # Market proxy: equal-weight of available
    mkt = rets.mean(axis=1).values

    # Vectorized rolling metrics
    roll_mean = rolling_mean_2d(rets.values, window)
    roll_std = rolling_std_2d(rets.values, window)
    ann_vol = roll_std * np.sqrt(252)
    ann_ret = roll_mean * 252
    sharpe = np.where(ann_vol > 1e-12, ann_ret / ann_vol, np.nan)
    beta = rolling_beta_2d(rets.values, mkt, window)
    cum_ret = np.nancumsum(rets.values, axis=0)
    max_dd = rolling_max_dd_2d(cum_ret, window)

    # Build results DataFrame
    rows = []
    for i, t in enumerate(valid_tickers):
        vol_col = ann_vol[:, i]
        valid_idx = np.where(~np.isnan(vol_col))[0]
        if len(valid_idx) == 0:
            continue
        last_idx = valid_idx[-1]
        rows.append({
            "universe": universe,
            "ticker": t,
            "window": window,
            "vol": float(ann_vol[last_idx, i]),
            "ann_ret": float(ann_ret[last_idx, i]),
            "sharpe": float(sharpe[last_idx, i]) if not np.isnan(sharpe[last_idx, i]) else np.nan,
            "max_dd": float(max_dd[last_idx, i]),
            "beta": float(beta[last_idx, i]) if not np.isnan(beta[last_idx, i]) else np.nan,
            "vol_stability": float(np.nanstd(roll_std[:, i])) if np.sum(~np.isnan(roll_std[:, i])) > 5 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("vol")
    print(f"=== Rolling {window}d metrics · {universe} ({len(df)} names) ===")
    print(df.head(15).to_string(index=False))
    if save:
        df.to_parquet(OUT)
        print(f"Wrote {OUT}")
        if checkpoint is not None:
            try:
                last_d = prices["date"].max()
                if hasattr(last_d, "date"):
                    last_d = last_d.date()
                checkpoint.mark_all_complete(list(df["ticker"].astype(str)), last_d)
            except Exception as e:
                print(f"checkpoint mark skip: {e}")

    # rolling dual-screen stability from history if present
    hist = DATA_DIR / "preferred_metrics_history.parquet"
    if hist.exists():
        h = pd.read_parquet(hist)
        h["as_of_date"] = pd.to_datetime(h["as_of_date"])
        g = h.groupby("ticker").agg(
            n_dates=("as_of_date", "count"),
            buffett_rate=("buffett_pass", "mean"),
            trifecta_rate=("trifecta_pass", "mean"),
            dual_rate=("decision", lambda s: (s == "INCLUDE_CORE").mean()),
            median_composite=("composite_score", "median"),
            composite_std=("composite_score", "std"),
        ).reset_index()
        g = g.sort_values("median_composite", ascending=False)
        g.to_parquet(OUT_STAB)
        print("\n=== Screen stability (through fundamentals history) ===")
        print(g.head(12).to_string(index=False))
        print(f"Wrote {OUT_STAB}")
    return df


def main():
    ap = argparse.ArgumentParser()
    add_index_args(ap, default="all")
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    uni = "all"
    if args.universe:
        uni = str(args.universe[-1])
    elif args.index:
        uni = str(args.index[-1])
    ck = None
    try:
        from resumable_job import JobCheckpoint
        ck = JobCheckpoint("rolling_window_analysis", "daily_prices")
        tickers = resolve(uni)
        if not ck.is_valid(tickers):
            print("rolling checkpoint invalid (prices or universe changed) — full recompute")
            ck._init_state(tickers)
            ck._save()
    except Exception as e:
        print(f"checkpoint unused: {e}")
        ck = None
    run(uni, args.window, save=args.save, checkpoint=ck)


if __name__ == "__main__":
    main()
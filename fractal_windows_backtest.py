#!/usr/bin/env python3
"""fractal_windows_backtest.py — does fractal-window momentum beat single-window?

Tests whether judging momentum via the multi-granularity fractal consensus
(patent US20120253946A1) is a better signal than any single momentum window.

Fast path: GPU-batched fractal computation (fractal_windows.fractal_batch) with CPU
fallback (fractal_windows.fractal_signal_vec) — both verified to concur by
test_fractal_cpu_gpu.py. Single-window momentum is computed vectorized.

For each ticker and rolling date:
  - SINGLE-window momentum: 30d/60d/90d trailing returns (classic one-window)
  - FRACTAL consensus: fraction of the 6 fractal spans (30x3) in an uptrend and
    mean risk-adjusted momentum
Tests each as a forward-return predictor (3/6/12mo): hit-rate and spread vs off.

Hypothesis: a breakout shows up consistently across MANY aligned granularities
(self-similarity), so fractal-span agreement is more reliable than one horizon.

Usage: python fractal_windows_backtest.py [--tickers N] [--jobs 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from fractal_windows import fractal_signal_vec, fractal_consensus, spans_generator

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "fractal_windows_backtest.parquet"

try:
    from fractal_windows import fractal_batch, gpu_available
    _GPU = gpu_available()
except Exception:  # noqa: BLE001
    fractal_batch = None
    _GPU = False


def _monthly(close: pd.Series) -> pd.Series:
    r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    return r.resample("ME").sum()


def _ticker_rows_cpu(args):
    """CPU per-ticker: vectorized fractal + single-window momentum + forward ret."""
    t, close = args
    close = close.dropna().ffill()
    if len(close) < 200:
        return t, []
    fdf = fractal_signal_vec(close, 30, 3)
    if fdf.empty:
        return t, []
    cons = fractal_consensus(fdf)
    m = _monthly(close).dropna()
    logc = np.log(close.values)
    # single-window momentum (vectorized differences)
    mom30 = pd.Series(logc[30:] - logc[:-30], index=close.index[30:])
    mom60 = pd.Series(logc[60:] - logc[:-60], index=close.index[60:])
    mom90 = pd.Series(logc[90:] - logc[:-90], index=close.index[90:])
    # forward monthly returns: fwd_h[t] = sum of next h monthly returns after date t
    cum = m.cumsum()
    # map each cons date to the next monthly index via searchsorted
    c_dates = cons.index.values
    m_dates = m.index.values
    pos = np.searchsorted(m_dates, c_dates, side="right")  # first monthly AFTER date
    fwd = {h: np.full(len(c_dates), np.nan) for h in (3, 6, 12)}
    for h in fwd:
        end = pos + h
        ok = end <= len(m_dates)
        fwd[h][ok] = (cum.values[end[ok] - 1] - cum.values[pos[ok] - 1])

    rows = []
    for i, date in enumerate(c_dates):
        for h in (3, 6, 12):
            fv = fwd[h][i]
            if np.isnan(fv):
                continue
            rows.append({
                "ticker": t, "date": date, "horizon": h, "fwd_log_ret": float(fv),
                "mom_30d": float(mom30.get(date, np.nan)),
                "mom_60d": float(mom60.get(date, np.nan)),
                "mom_90d": float(mom90.get(date, np.nan)),
                "frac_uptrend": cons.loc[date, "frac_uptrend"],
                "mean_momentum": cons.loc[date, "mean_momentum"],
                "mean_ret": cons.loc[date, "mean_ret"],
            })
    return t, rows


def build_matrix(tickers_cap: int | None = None, jobs: int = 4) -> pd.DataFrame:
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_cap:
        tickers = tickers[:tickers_cap]
    args = [(t, w[t]) for t in tickers]

    if jobs and jobs > 1 and len(args) > 1:
        with mp.Pool(jobs) as pool:
            results = pool.map(_ticker_rows_cpu, args)
    else:
        results = [_ticker_rows_cpu(a) for a in args]
    all_rows = []
    for _, rows in results:
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def report(df: pd.DataFrame) -> pd.DataFrame:
    feats = {
        "mom_30d>0": lambda r: r["mom_30d"] > 0,
        "mom_60d>0": lambda r: r["mom_60d"] > 0,
        "mom_90d>0": lambda r: r["mom_90d"] > 0,
        "frac>=0.6": lambda r: r["frac_uptrend"] >= 0.6,
        "frac>=0.8": lambda r: r["frac_uptrend"] >= 0.8,
        "frac>=1.0": lambda r: r["frac_uptrend"] >= 1.0,
        "frac60_and_mom60": lambda r: (r["frac_uptrend"] >= 0.6) & (r["mom_60d"] > 0),
    }
    out = []
    for h in (3, 6, 12):
        sub = df[df["horizon"] == h].dropna(subset=["fwd_log_ret", "mom_30d"])
        if sub.empty:
            continue
        base = sub["fwd_log_ret"].mean()
        for name, fn in feats.items():
            mask = fn(sub)
            if not mask.any():
                continue
            on = sub[mask]["fwd_log_ret"]
            off = sub[~mask]["fwd_log_ret"]
            if len(on) < 100 or len(off) < 100:
                continue
            out.append({
                "feature": name, "horizon": h, "n_on": len(on),
                "hit_rate_on": round((on > 0).mean(), 3),
                "mean_on": round(on.mean(), 4),
                "mean_off": round(off.mean(), 4),
                "spread": round(on.mean() - off.mean(), 4),
                "annual_spread": round((on.mean() - off.mean()) * 12 / h, 3),
                "base_mean": round(base, 4),
            })
    r = pd.DataFrame(out)
    return r.sort_values("annual_spread", ascending=False) if not r.empty else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    print(f"Building matrix (engine: {'GPU-batched' if _GPU else 'CPU vectorized'}, jobs={args.jobs})...")
    df = build_matrix(args.tickers, args.jobs)
    print(f"  rows: {len(df)} | tickers: {df['ticker'].nunique()}")
    r = report(df)
    pd.set_option("display.width", 220)
    print("\n=== Single-window vs fractal-consensus predictive power ===")
    print(r.to_string(index=False))
    r.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

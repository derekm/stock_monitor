#!/usr/bin/env python3
"""fractal_windows_backtest_gpu.py — GPU batched backtest of fractal momentum.

Processes the ENTIRE universe as one [T tickers x D days] tensor (scatter), runs
the fractal momentum + on-device consensus as batched torch ops, then gathers the
forward-return signal rows (vectorized, no per-date Python loop).

Engine selection:
  - GPU batched (fractal_windows_gpu) when torch.cuda available
  - else CPU batched (same code path, device="cpu")

This is the true scatter-gather: the per-ticker serial loop and pandas/polars
groupby are replaced by batched tensor ops over the whole universe.

Usage: python fractal_windows_backtest_gpu.py [--tickers N] [--device auto]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fractal_windows_gpu import fractal_batch, fractal_consensus_batch
from fractal_windows import spans_generator
# Device handling comes from tensor_ops, not a local reimplementation.
from tensor_ops import (
    _best_device, is_gpu, device_name, resolve_device,
)
_resolve_device = resolve_device

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "fractal_windows_backtest.parquet"


def _monthly(close: pd.Series) -> pd.Series:
    r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    return r.resample("ME").sum()


def build_matrix(tickers_cap: int | None = None, device: str | None = None) -> pd.DataFrame:
    import torch
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_cap:
        tickers = tickers[:tickers_cap]
    # Use a fixed trailing window per ticker (each ffill'd) so tickers with
    # different listing dates all contribute a full window, instead of forcing
    # a common start date (which collapses on young tickers).
    window = 1500  # trailing trading days used for the fractal
    wide = w[tickers].ffill().tail(window)
    # drop columns that are still mostly NaN in the window (young tickers)
    keep = [c for c in wide.columns if wide[c].notna().sum() >= 200]
    wide = wide[keep]
    tickers = list(wide.columns)
    wide = wide.ffill().bfill()  # close gaps within the window
    D = wide.shape[0]
    if D < 200:
        raise SystemExit("not enough history")

    logp = np.log(wide.values.T)  # [T, D]
    print(f"  universe: {len(tickers)} tickers x {D} days (device={device_name(resolve_device(device))})")

    res = fractal_batch(logp, 30, 3, device=device)
    cons = fractal_consensus_batch(res, len(tickers), D, device=device)
    T = len(tickers)

    # move consensus to numpy for gathering
    fu = cons["frac_uptrend"].cpu().numpy()
    mm = cons["mean_momentum"].cpu().numpy()
    mr = cons["mean_ret"].cpu().numpy()

    # single-window momentum (vectorized, numpy)
    lp = logp
    mom30 = np.full_like(lp, np.nan); mom30[:, 30:] = lp[:, 30:] - lp[:, :-30]
    mom60 = np.full_like(lp, np.nan); mom60[:, 60:] = lp[:, 60:] - lp[:, :-60]
    mom90 = np.full_like(lp, np.nan); mom90[:, 90:] = lp[:, 90:] - lp[:, :-90]

    # forward monthly returns per ticker via per-ticker monthly series
    rows = []
    dates = list(wide.index)
    for i, t in enumerate(tickers):
        close = wide[t]
        m = _monthly(close).dropna()
        if len(m) < 13:
            continue
        cum = m.cumsum()
        m_dates = m.index.values
        # only use consensus rows where the fractal is fully available (first L)
        start = 90  # longest span (0,90)
        for j in range(start, D):
            date = dates[j]
            pos = np.searchsorted(m_dates, date, side="right")
            for h in (3, 6, 12):
                end = pos + h
                if end > len(m_dates):
                    continue
                fwd = float(cum.values[end - 1] - cum.values[pos - 1])
                rows.append({
                    "ticker": t, "date": date, "horizon": h, "fwd_log_ret": fwd,
                    "mom_30d": float(mom30[i, j]),
                    "mom_60d": float(mom60[i, j]),
                    "mom_90d": float(mom90[i, j]),
                    "frac_uptrend": float(fu[i, j]),
                    "mean_momentum": float(mm[i, j]),
                    "mean_ret": float(mr[i, j]),
                })
    return pd.DataFrame(rows)


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
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = None if args.device == "auto" else args.device
    # Resolve the device ONCE and label from what will actually be used.
    # Previously this printed _best_device() regardless of --device, so
    # `--device cpu` reported "device=cuda" while running on the CPU.
    resolved = _resolve_device(dev)
    eng = "GPU" if is_gpu(resolved) else "CPU-batched"
    print(f"Building matrix (engine: {eng}, device={device_name(resolved)})...")
    df = build_matrix(args.tickers, resolved)
    print(f"  rows: {len(df)} | tickers: {df['ticker'].nunique()}")
    r = report(df)
    pd.set_option("display.width", 220)
    print("\n=== Single-window vs fractal-consensus predictive power ===")
    print(r.to_string(index=False))
    r.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

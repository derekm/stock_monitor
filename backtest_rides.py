#!/usr/bin/env python3
"""backtest_rides.py — historical backtest of ride-entry/exit approaches.

Compares four ride strategies on a universe of tickers with enough history,
using only data available at each monthly decision date (no lookahead):

  A. classic         — enter mom12>thresh & mom3>0, exit mom3<=0 (baseline)
  B. quality_gate    — ride_gate entry (no 12mo need, uses fractal stack +
                       durability), classic exit
  C. dual_exit       — classic entry, ride_exit (dual-condition + trailing stop)
  D. quality_dual    — ride_gate entry + ride_exit (the full new approach)

For each ticker we precompute the full HISTORICAL series of: monthly log
returns, the fractal momentum-stack depth (per date), and the long-ride
durability score (per date), then walk forward month by month applying each
strategy's rule. Positions shift 1 month (signal->trade lag, no lookahead).

Outputs a comparison table of per-strategy total return, mean per-ticker
return, hit rate (tickers where strategy beats buy-hold), n trades, max drawdown.
Saves a parquet of per-ticker results.

Usage: python backtest_rides.py [--n 120] [--entry 0.40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fractal_windows import fractal_multi_view, momentum_stack_series
from ride_longevity import long_ride_score, ride_gate, ride_exit

DATA_DIR = ROOT
OUT = DATA_DIR / "backtest_rides.parquet"
MIN_MONTHS = 36


def load_prices() -> pd.DataFrame:
    px = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                         columns=["date", "ticker", "close", "volume"])
    px["date"] = pd.to_datetime(px["date"])
    return px


def monthly_returns(close: pd.Series) -> pd.Series:
    m = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return m.resample("ME").sum().dropna()


def precompute(close: pd.Series, volume: pd.Series | None):
    """Return dict of full historical monthly-aligned signal series."""
    m = monthly_returns(close)
    if len(m) < 3:
        return None
    # fractal stack depth per date (daily), resampled to month-end
    try:
        mv = fractal_multi_view(close, configs=[(5, 3), (10, 3), (15, 3), (30, 3)])
        stk = momentum_stack_series(mv)["stack_depth"]
        stk = stk.resample("ME").last()
        # long-ride durability per date, resampled
        lr = long_ride_score(close, volume)["long_ride_score"] if volume is not None \
            else long_ride_score(close, None)["long_ride_score"]
        lr = lr.resample("ME").last()
    except Exception:
        return None
    # align to monthly index
    idx = m.index
    stack = stk.reindex(idx).ffill().fillna(0).astype(int)
    dur = lr.reindex(idx).ffill().fillna(0.0)
    return {"m": m, "stack": stack, "long_ride": dur}


def simulate(ticker: str, close: pd.Series, volume: pd.Series | None,
             pre, entry_thresh: float) -> dict:
    """Run all four strategies on one ticker; return per-strategy stats."""
    m = pre["m"]
    cum = m.cumsum()
    mom12 = m.rolling(12, min_periods=1).sum()
    mom3 = m.rolling(3, min_periods=1).sum()
    n = len(m)
    results = {}
    strategies = {
        "classic":       {"entry": "classic", "exit": "classic"},
        "quality_gate":  {"entry": "quality", "exit": "classic"},
        "dual_exit":     {"entry": "classic", "exit": "dual"},
        "quality_dual":  {"entry": "quality", "exit": "dual"},
        "quality_dual_persist": {"entry": "quality", "exit": "dual", "persist": True},
    }
    for name, cfg in strategies.items():
        pos = np.zeros(n)
        in_ride = False
        rollover_streak = 0
        for i in range(n):
            mom12i = float(mom12.iloc[i]) if pd.notna(mom12.iloc[i]) else np.nan
            mom3i = float(mom3.iloc[i]) if pd.notna(mom3.iloc[i]) else np.nan
            stack = int(pre["stack"].iloc[i])
            dur = float(pre["long_ride"].iloc[i])
            if not in_ride:
                if cfg["entry"] == "classic":
                    enter = (pd.notna(mom12i) and mom12i > entry_thresh and
                             pd.notna(mom3i) and mom3i > 0)
                else:
                    rg = ride_gate(m.iloc[: i + 1], entry_thresh=entry_thresh,
                                   stack_depth=stack, long_ride=dur)
                    enter = rg["gate_open"]
                if enter:
                    in_ride = True
                    rollover_streak = 0
            else:
                if cfg["exit"] == "classic":
                    exit_now = pd.notna(mom3i) and mom3i <= 0
                    if exit_now:
                        rollover_streak = 0
                else:
                    # dual exit: get confirm/mom3; persistence handled here
                    ex = ride_exit(m.iloc[: i + 1], stack_depth=stack,
                                   long_ride=dur, trailing_stop=-0.25, persist=99)
                    # track consecutive rollover+confirm months
                    roll = (ex["mom3"] is not None and ex["mom3"] <= 0 and ex["confirm"])
                    streak_need = 2 if cfg.get("persist") else 1
                    rollover_streak = rollover_streak + 1 if roll else 0
                    exit_now = ex["exit"] or rollover_streak >= streak_need
                if exit_now:
                    in_ride = False
                    rollover_streak = 0
            # shift position by 1 (signal at month i -> position month i+1)
            if i >= 1:
                pos[i] = 1.0 if in_ride else 0.0
        ride = float((m.values * pos).sum())
        bh = float(m.sum())
        eq = (1 + m.values * pos).cumprod()
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        bh_eq = (1 + m.values).cumprod()
        bh_dd = float((bh_eq / np.maximum.accumulate(bh_eq) - 1).min())
        results[name] = {
            "ride_return": ride, "buy_hold": bh, "excess": ride - bh,
            "max_dd_ride": dd, "max_dd_bh": bh_dd,
            "n_trades": int(np.sum(np.abs(np.diff(pos)) > 0.5)),
            "in_market": float(pos.mean()),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="max tickers to backtest")
    ap.add_argument("--entry", type=float, default=0.40)
    args = ap.parse_args()

    px = load_prices()
    # universe: tickers with >= MIN_MONTHS history, take the most-traded sample
    lens = px.groupby("ticker")["date"].nunique()
    candidates = lens[lens >= MIN_MONTHS * 21].index.tolist()
    # prioritize ride-long + fresh-breakout names from the current run
    try:
        cur = pd.read_parquet(DATA_DIR / "shock_ride_tickers.parquet")
        priority = set(cur[(cur["ride_long"] == 1) | (cur["fresh_verdict"] == "FRESH_BREAKOUT")]["ticker"])
    except Exception:
        priority = set()
    ordered = sorted(candidates, key=lambda t: (t not in priority, t))
    ordered = ordered[: args.n]
    print(f"Backtesting {len(ordered)} tickers (>= {MIN_MONTHS}mo history), entry {args.entry:.0%}")

    # load full price matrix for fast per-ticker slicing
    w = px.pivot(index="date", columns="ticker", values="close")
    vm = px.pivot(index="date", columns="ticker", values="volume")

    agg = {k: {"ride_return": [], "buy_hold": [], "excess": [], "max_dd_ride": [],
               "max_dd_bh": [], "n_trades": [], "in_market": []}
           for k in ["classic", "quality_gate", "dual_exit", "quality_dual",
                     "quality_dual_persist"]}
    per_ticker_rows = []
    for tkr in ordered:
        if tkr not in w.columns:
            continue
        close = w[tkr].dropna()
        vol = vm[tkr].dropna() if tkr in vm.columns else None
        pre = precompute(close, vol)
        if pre is None or len(pre["m"]) < 3:
            continue
        res = simulate(tkr, close, vol, pre, args.entry)
        for k, stats in res.items():
            for key, val in stats.items():
                agg[k][key].append(val)
        per_ticker_rows.append({"ticker": tkr, **{f"{k}_{key}": v for k, st in res.items() for key, v in st.items()}})

    rows = []
    for k, d in agg.items():
        a = pd.DataFrame(d)
        rows.append({
            "strategy": k,
            "total_ride_return": round(float(a["ride_return"].sum()), 3),
            "total_buy_hold": round(float(a["buy_hold"].sum()), 3),
            "mean_excess": round(float(a["excess"].mean()), 4),
            "median_excess": round(float(a["excess"].median()), 4),
            "hit_rate": round(float((a["excess"] > 0).mean()), 3),
            "mean_max_dd": round(float(a["max_dd_ride"].mean()), 4),
            "total_trades": int(a["n_trades"].sum()),
            "mean_in_market": round(float(a["in_market"].mean()), 3),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== STRATEGY COMPARISON (sum across tickers, signal->trade lag 1mo) ===")
    print(out.to_string(index=False))
    print("\n(buy-hold is the same across strategies per ticker; excess is ride-buyhold)")

    per = pd.DataFrame(per_ticker_rows)
    if len(per):
        per.to_parquet(OUT, index=False)
        print(f"\nWrote per-ticker backtest to {OUT} ({len(per)} tickers)")
    return 0


if __name__ == "__main__":
    exit(main())

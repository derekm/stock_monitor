#!/usr/bin/env python3
"""backtest_structural.py — daily-granularity backtest of structural / risk-scaled
gate paradigms against the momentum-threshold ride gate.

The ride gate (ride_longevity.ride_gate) is a LAGGING level detector: it reads
"momentum > 0.40" after a surge, buys the top, then holds through the pullback
(the young path suppresses the exit). On volatile / whipsaw names (e.g. RAL) it
loses to buy-hold. This script backtests FOUR fundamentally different paradigms
plus two hybrids against the momentum gate, all at daily frequency with NO
lookahead, using the SAME structural_positions() engine that feeds the live gate:

  1) turtle     — Donchian 55-day breakout entry + 2x ATR chandelier trailing stop
  2) volscale   — exposure sized to target annualized vol, gated by SMA200 trend
  3) regime     — EMA50/EMA200 markup/distribution state machine
  4) recouple   — enter when close re-couples above EMA21 AND EMA50, size by 1/vol
  5) momentum   — classic daily momentum gate (baseline, for comparison)
  6) hybrid     — momentum entry + vol-scaled size + 2x ATR chandelier stop
  7) consensus  — majority of the four structural signals, vol-scaled size

Position is decided using only data up to the prior close and applied to the
next trading day (no lookahead). Results compare each paradigm to buy-and-hold.

Outputs:
  backtest_structural.parquet — per-ticker per-paradigm returns / drawdown / in-mkt
Usage: python backtest_structural.py [--n 250]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ride_longevity import structural_positions, STRUCTURAL_MODES

DATA_DIR = ROOT
OUT = DATA_DIR / "backtest_structural.parquet"
MIN_DAYS = 60
MODES = list(STRUCTURAL_MODES)


def load_prices():
    px = pd.read_parquet(DATA_DIR / "daily_prices/",
                         columns=["date", "ticker", "close"])
    px["date"] = pd.to_datetime(px["date"])
    return px


def simulate_ticker(close: pd.Series) -> dict:
    """For each mode: positions decided at prior close, applied next day."""
    ret = close.pct_change().fillna(0.0).to_numpy()
    bh_eq = (1 + ret).cumprod()
    bh_dd = float((bh_eq / np.maximum.accumulate(bh_eq) - 1).min())
    results = {}
    for mode in MODES:
        p = structural_positions(close, mode=mode).to_numpy()
        p_prev = np.roll(p, 1); p_prev[0] = 0.0
        r = ret * p_prev
        eq = (1 + r).cumprod()
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        results[mode] = {
            "ride_return": float(r.sum()),
            "buy_hold": float(ret.sum()),
            "excess": float(r.sum() - ret.sum()),
            "max_dd_ride": dd,
            "max_dd_bh": bh_dd,
            "in_market": float(p_prev.mean()),
        }
    results["buy_hold"] = {"ride_return": float(ret.sum()), "buy_hold": float(ret.sum()),
                           "excess": 0.0, "max_dd_ride": bh_dd, "max_dd_bh": bh_dd,
                           "in_market": 1.0}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    px = load_prices()
    lens = px.groupby("ticker")["date"].nunique()
    candidates = lens[lens >= MIN_DAYS * 3].index.tolist()
    candidates = candidates[: args.n]

    w = px.pivot(index="date", columns="ticker", values="close")
    keys = MODES + ["buy_hold"]
    agg = {k: {"ride_return": [], "buy_hold": [], "excess": [], "max_dd_ride": [],
               "max_dd_bh": [], "in_market": []} for k in keys}
    per_rows = []
    for tkr in candidates:
        if tkr not in w.columns:
            continue
        close = w[tkr].dropna()
        if len(close) < 60:
            continue
        res = simulate_ticker(close)
        for k, st in res.items():
            for key, val in st.items():
                agg[k][key].append(val)
        per_rows.append({"ticker": tkr, **{f"{k}_{key}": v for k, st in res.items() for key, v in st.items()}})

    rows = []
    for k in keys:
        a = pd.DataFrame(agg[k])
        calmar = float(a["ride_return"].sum() / abs(a["max_dd_ride"].mean())) if a["max_dd_ride"].mean() < 0 else np.nan
        rows.append({
            "paradigm": k,
            "total_ride_return": round(float(a["ride_return"].sum()), 3),
            "total_buy_hold": round(float(a["buy_hold"].sum()), 3),
            "mean_excess": round(float(a["excess"].mean()), 4),
            "median_excess": round(float(a["excess"].median()), 4),
            "hit_rate": round(float((a["excess"] > 0).mean()), 3),
            "mean_max_dd": round(float(a["max_dd_ride"].mean()), 4),
            "calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
            "mean_in_market": round(float(a["in_market"].mean()), 3),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n=== STRUCTURAL PARADIGMS (daily, {len(candidates)} tickers, no lookahead) ===")
    print(out.to_string(index=False))
    print("\n(excess = paradigm - buy-hold per ticker; total is summed across tickers)")

    per = pd.DataFrame(per_rows)
    if len(per):
        per.to_parquet(OUT, index=False)
        print(f"\nWrote per-ticker results to {OUT} ({len(per)} tickers)")
    return 0


if __name__ == "__main__":
    exit(main())
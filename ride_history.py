#!/usr/bin/env python3
"""
ride_history.py — point-in-time recommended ride trade history per ticker.

Answers "what would the ride rule have said each month for <ticker>?" with NO
lookahead — recomputing momentum, fractal stack, ride gate, and dual exit at
every monthly step exactly as shock_ride.py does, then recording the
recommendation the rule would have given then.

Use this to audit a name's ride signals over time (entry / exit / hold) rather
than trusting only the latest shock_ride snapshot.

Method (faithful to shock_ride.py):
  - monthly LOG returns via `resample('ME').sum()` (same as _monthly_returns)
  - mom3 = trailing 3-month sum, mom12 = trailing 12-month sum
  - fractal 4-view stack (15d/30d/45d/90d) + posture on the daily close up to
    that month's end
  - long_ride_score (durability) on the daily series up to that month's end
  - ride_gate (quality entry, no 12mo history requirement) + ride_exit
    (dual-condition exit, trailing_stop=-0.25)
  - recommendation via the SAME young/established branch: a ticker with
    < MIN_TICKER_HISTORY (36) months is "young" -> BUY iff gate open (exit
    ignored); otherwise the full dual-exit logic applies.

Outputs:
  ride_history.csv / ride_history.parquet — as_of, ticker, n_months,
  established, mom3, mom12, posture, stack_depth, long_ride_score,
  ride_gate_open, gate_horizon, gate_mom, ride_exit_flag, exit_kind,
  recommendation

Usage: python ride_history.py --ticker RAL [--ticker NVDA ...] [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR
from fractal_windows import fractal_multi_view, fractal_posture, momentum_stack
from ride_longevity import ride_gate, ride_exit, long_ride_score
from momentum_research import research_report

OUT_CSV = DATA_DIR / "ride_history.csv"
OUT_PQ = DATA_DIR / "ride_history.parquet"

ENTRY_THRESH = 0.40
MIN_TICKER_HISTORY = 36  # months of price history required for a full ride
CONFIGS = [(5, 3), (10, 3), (15, 3), (30, 3)]


def _volume_series(ticker: str) -> pd.Series | None:
    try:
        v = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                            columns=["date", "ticker", "volume"])
        v["date"] = pd.to_datetime(v["date"])
        vm = v.pivot_table(index="date", columns="ticker", values="volume").sort_index().ffill()
        if ticker in vm.columns:
            return vm[ticker].dropna()
    except Exception:
        pass
    return None


def ride_history_for(ticker: str) -> pd.DataFrame:
    """Point-in-time ride signal history for one ticker (no lookahead)."""
    prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                             columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    s = (prices[prices["ticker"] == ticker].set_index("date")["close"]
         .sort_index().dropna())
    if len(s) < 30:
        return pd.DataFrame()

    vol_ser = _volume_series(ticker)

    logret = np.log(s / s.shift(1))
    mret = logret.resample("ME").sum().replace([np.inf, -np.inf], np.nan).dropna()

    rows = []
    for i in range(2, len(mret)):
        hist_m = mret.iloc[:i + 1]
        mom3 = hist_m.tail(3).sum()
        mom12 = hist_m.tail(12).sum() if len(hist_m) >= 12 else np.nan
        cutoff = hist_m.index[-1]
        s_part = s[s.index <= cutoff]
        if len(s_part) < 30:
            continue

        mv = fractal_multi_view(s_part, configs=CONFIGS)
        p = fractal_posture(mv)
        stack = momentum_stack(mv)
        stack_depth = int(stack.get("stack_depth", 0))

        lr = long_ride_score(s_part, vol_ser[s_part.index]) if vol_ser is not None else None
        lr_val = float(lr["long_ride_score"].iloc[-1]) if lr is not None and len(lr) else 0.0

        established = len(hist_m) >= MIN_TICKER_HISTORY

        try:
            rr = research_report(hist_m, annual_vol=hist_m.tail(12).std() * np.sqrt(12))
            yg = rr["young_gate"]
        except Exception:
            yg = {"gate_open": False, "reliability": "low"}

        rg = ride_gate(hist_m, entry_thresh=ENTRY_THRESH, stack_depth=stack_depth,
                       long_ride=lr_val, reliability=yg["reliability"])
        ex = ride_exit(hist_m, stack_depth=stack_depth, long_ride=lr_val,
                       trailing_stop=-0.25)

        if not established:
            rec = "BUY" if rg["gate_open"] else "FLAT"
        elif rg["gate_open"] and ex["exit"]:
            rec = "AVOID/EXIT"
        elif rg["gate_open"]:
            rec = "BUY"
        elif pd.notna(mom12) and mom12 > 0.40 and mom3 <= 0:
            rec = "AVOID"
        elif pd.notna(mom12) and mom12 > 0.40:
            rec = "WATCH"
        else:
            rec = "FLAT"

        rows.append({
            "as_of": hist_m.index[-1].strftime("%Y-%m-%d"),
            "ticker": ticker,
            "n_months": len(hist_m),
            "established": int(established),
            "mom3": round(float(mom3), 4),
            "mom12": round(float(mom12), 4) if pd.notna(mom12) else np.nan,
            "posture": p["posture"],
            "stack_depth": stack_depth,
            "long_ride_score": round(lr_val, 4),
            "ride_gate_open": int(rg["gate_open"]),
            "gate_horizon": rg["horizon"],
            "gate_mom": round(float(rg["mom_used"]), 4) if pd.notna(rg["mom_used"]) else np.nan,
            "ride_exit_flag": int(ex["exit"]),
            "exit_kind": ex["exit_kind"],
            "recommendation": rec,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", required=True,
                    help="Comma-separated tickers to reconstruct (e.g. RAL,NVDA)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    frames = []
    for t in tickers:
        h = ride_history_for(t)
        if h.empty:
            print(f"{t}: no data (< 30 daily closes)")
            continue
        frames.append(h)
        print(f"{t}: {len(h)} monthly ride signals")

    if not frames:
        return 1
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "as_of"])

    if args.save:
        out.to_csv(OUT_CSV, index=False)
        out.to_parquet(OUT_PQ, index=False)
        print(f"Wrote {OUT_CSV} / {OUT_PQ} ({len(out)} rows)")

    for t in tickers:
        sub = out[out["ticker"] == t]
        if sub.empty:
            continue
        print(f"\n=== {t} — ride trade history ===")
        pd.set_option("display.width", 220)
        show = sub[["as_of", "n_months", "established", "mom3", "mom12", "posture",
                    "stack_depth", "long_ride_score", "ride_gate_open", "gate_horizon",
                    "ride_exit_flag", "exit_kind", "recommendation"]]
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
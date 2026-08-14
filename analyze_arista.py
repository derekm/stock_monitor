#!/usr/bin/env python3
"""analyze_arista.py — ARISTA reliability & short-placement study.

Honest historical answers to:
  1. How reliable is arista_signal? (fraction that precede a >=15% drawdown,
     as a function of score).
  2. Does score select intensity? (caught rate rises with score?)
  3. Can it time a SHORT (day-of vs week-off)? The signal leads the trough by
     ~56d, so a fixed 21/63d short loses; test entry-lag variants that wait
     for breakdown confirmation.

Reads arista_metrics.parquet + daily_prices.parquet. Point-in-time only.
Outputs arista_reliability.parquet (per-signal forward stats).

Usage:
  python analyze_arista.py reliability   # score-bucket caught rates + timing
  python analyze_arista.py short         # short-placement variants
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
WINDOW = 120

DF = None
PX = None


def _load():
    global DF, PX
    DF = pd.read_parquet(DATA_DIR / "arista_metrics.parquet")
    PX = pd.read_parquet(DATA_DIR / "daily_prices.parquet")
    PX["date"] = pd.to_datetime(PX["date"])


def _fwd_stats(close: np.ndarray, window: int):
    n = len(close)
    rev = close[::-1].copy()
    fmax = pd.Series(rev).rolling(window, min_periods=1).max().to_numpy()[::-1]
    fmin = pd.Series(rev).rolling(window, min_periods=1).min().to_numpy()[::-1]
    dd = np.where(fmax > 0, fmin / fmax - 1, np.nan)
    return dd, fmin, fmax


def _signal_rows(filter_extra=None) -> pd.DataFrame:
    rows = []
    for tk, g in DF.groupby("ticker"):
        cs = PX[PX["ticker"] == tk].sort_values("date")
        if cs.empty:
            continue
        c = cs["close"].to_numpy()
        dates = cs["date"].to_numpy(dtype="datetime64[ns]")
        if len(c) < WINDOW:
            continue
        dd, fmin, fmax = _fwd_stats(c, WINDOW)
        g = g.sort_values("date")
        sig = g[g["arista_signal"]]
        if filter_extra:
            sig = sig[filter_extra(sig)]
        if sig.empty:
            continue
        idx = np.searchsorted(dates, sig["date"].to_numpy(dtype="datetime64[ns]"), side="right")
        for p, (_, srow) in zip(idx, sig.iterrows()):
            if p >= len(c):
                continue
            mdd = dd[p]
            if np.isnan(mdd):
                continue
            seg = c[p:p + WINDOW]
            tmin = p + int(np.nanargmin(seg))
            d2t = int((dates[min(tmin, len(dates) - 1)] - dates[p]) / np.timedelta64(1, "D"))
            p5 = min(p + 5, len(c) - 1)
            rows.append({
                "ticker": tk, "date": srow["date"],
                "score": float(srow["arista_score"]), "decel": float(srow["decel"]),
                "downshare": float(srow["downshare"]), "from20": float(srow["from20"]),
                "fwd_max_dd": float(mdd), "caught15": bool(mdd <= -0.15),
                "days_to_trough": d2t,
                "short_dayof": (c[p] - c[tmin]) / c[p] if tmin < len(c) else np.nan,
                "short_weekoff": (c[p5] - c[tmin]) / c[p5] if tmin > p5 else np.nan,
            })
    R = pd.DataFrame(rows)
    R.to_parquet(DATA_DIR / "arista_reliability.parquet", index=False)
    return R


def cmd_reliability(_args):
    R = _signal_rows()
    print(f"=== ARISTA reliability ({len(R)} signals over full universe) ===")
    print(f"OVERALL: caught>=15% dd {R['caught15'].mean():.0%} | avg fwd maxDD "
          f"{R['fwd_max_dd'].mean():+.0%} | median days-to-trough {R['days_to_trough'].median():.0f}")
    print("\n--- By score bucket ---")
    bins = [0, 0.2, 0.35, 0.5, 0.65, 1.01]
    labs = ["0-0.2", "0.2-0.35", "0.35-0.5", "0.5-0.65", "0.65+"]
    R["b"] = pd.cut(R["score"], bins, labels=labs)
    g = R.groupby("b", observed=True).agg(
        n=("caught15", "size"), caught=("caught15", "mean"),
        avg_dd=("fwd_max_dd", "mean"), med_days=("days_to_trough", "median"),
        dayof=("short_dayof", "mean"), weekoff=("short_weekoff", "mean"))
    print(g.round(3).to_string())
    print("\n--- Timing day-of vs week-off (mean gain to trough) ---")
    only = R[R["short_weekoff"].notna()]
    print(f"  day-of {R['short_dayof'].mean()*100:+.2f}% | week-off "
          f"{R['short_weekoff'].mean()*100:+.2f}% | paired diff "
          f"{((only['short_dayof']-only['short_weekoff']).mean())*100:+.2f}pp (n={len(only)})")
    print("\n--- High-confidence (score>=0.5) ---")
    hi = R[R["score"] >= 0.5]
    print(f"  n={len(hi)} | caught {hi['caught15'].mean():.0%} | avg fwd maxDD "
          f"{hi['fwd_max_dd'].mean():+.0%} | day-of {hi['short_dayof'].mean()*100:+.1f}% | "
          f"week-off {hi['short_weekoff'].mean()*100:+.1f}% | med d2t {hi['days_to_trough'].median():.0f}")
    return 0


def cmd_short(_args):
    R = _signal_rows(lambda s: (s["from20"] < -0.03) & (s["arista_score"] >= 0.5))
    print(f"=== ARISTA short-placement (signal + rollover + score>=0.5) n={len(R)} ===")
    print(f"caught>=15% {R['caught15'].mean():.0%} | avg fwd maxDD "
          f"{R['fwd_max_dd'].mean():+.0%} | day-of-to-trough {R['short_dayof'].mean()*100:+.1f}% "
          f"(win {(R['short_dayof']>0).mean():.0%}) | week-off {R['short_weekoff'].mean()*100:+.1f}% "
          f"| med d2t {R['days_to_trough'].median():.0f}")
    print("\nNOTE: raw signal leads trough ~56d, so a fixed-horizon short loses. "
          "This confirms ARISTA is a de-risk/exit trigger, not a short-entry timer.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("reliability").set_defaults(cmd="reliability")
    sub.add_parser("short").set_defaults(cmd="short")
    args = ap.parse_args()
    _load()
    if args.cmd == "reliability":
        return cmd_reliability(args)
    if args.cmd == "short":
        return cmd_short(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

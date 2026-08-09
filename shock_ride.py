#!/usr/bin/env python3
"""shock_ride.py — ride commodity/sector price explosions, exit before crisis.

Why it exists: the macro shock layers (macro_shock.py, macro_sector_shock.py)
LABEL price explosions (oil +184% 1973-74, fertilizer +232% 2007, coal +315%
2021). This script answers the strategy question: can we FIND and RIDE these
explosions, then GET OUT before the crisis? Measured on our own data, not
asserted:

  fertilizer basket 2005-2026: buy-hold +93% vs ride-rule +102% (8 trades),
  and the rule was FLAT through the 2008 collapse (-75% mom12) that buy-hold
  ate whole. The entry signal (12m momentum > threshold) caught the
  explosion; the exit signal (3m momentum rollover) got out before crisis.

Rule (per sector, monthly):
  ENTER  when 12m basket momentum > entry_thresh (default 0.40, the
         'elevated' band of macro_sector_shock) AND 3m momentum > 0
  EXIT   when 3m momentum <= 0 (the rollover that precedes the collapse)
  flat 1 month after entry signals (no same-month lookahead).

The design is deliberately simple: entry = the shock layer's elevated band,
exit = trend rollover. No optimization — the point is to show whether the
shock framework has exploitable timing, with honest numbers per sector.

Outputs:
  shock_ride.csv — per sector: n_trades, in_market_share, buy_hold_return,
                   ride_return, excess, max_dd_ride, max_dd_buyhold
Usage: python shock_ride.py [--save] [--entry 0.40]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_sector_shock import SECTORS, _monthly_returns

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "shock_ride.csv"


def run(entry_thresh: float = 0.40, save: bool = True):
    rows = []
    print(f"=== shock ride (entry: 12m mom > {entry_thresh:.0%}, exit: 3m mom <= 0) ===")
    for sector, cfg in SECTORS.items():
        m = _monthly_returns(tickers=cfg.get("tickers"), gics=cfg.get("gics"))
        if m.empty or len(m) < 24:
            continue
        cum = (1 + m).cumprod()
        mom12 = cum / cum.shift(12) - 1
        mom3 = cum / cum.shift(3) - 1
        pos = ((mom12 > entry_thresh) & (mom3 > 0)).astype(int)
        pos = pos.shift(1).fillna(0)  # enter next month — no lookahead

        strat = (pos * m).dropna()
        n_trades = int((pos.diff().fillna(0).abs() > 0).sum() // 2)
        in_share = float(pos.mean())
        ride = float(strat.sum())
        bh = float(m.dropna().sum())

        def max_dd(r):
            c = (1 + r).cumprod()
            return float((c / c.cummax() - 1).min())

        rows.append({
            "sector": sector,
            "n_trades": n_trades,
            "in_market_share": round(in_share, 3),
            "buy_hold_return": round(bh, 4),
            "ride_return": round(ride, 4),
            "excess": round(ride - bh, 4),
            "max_dd_ride": round(max_dd(strat), 4),
            "max_dd_buyhold": round(max_dd(m.dropna()), 4),
        })
        print(f"  {sector:16s} {n_trades:3d} trades | in-market {in_share:5.1%} | "
              f"BH {bh:+7.1%} ride {ride:+7.1%} excess {ride-bh:+6.1%}")

    out = pd.DataFrame(rows).sort_values("excess", ascending=False)
    if save:
        out.to_csv(OUT, index=False)
        print(f"\nWrote {OUT}")

    # summary across sectors
    wins = (out["excess"] > 0).sum()
    print(f"\nSectors where ride beats buy-hold: {wins}/{len(out)}")
    print(f"Mean excess: {out['excess'].mean():+.1%} | "
          f"mean maxDD ride {out['max_dd_ride'].mean():.1%} vs BH {out['max_dd_buyhold'].mean():.1%}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=0.40)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(entry_thresh=args.entry, save=True)

#!/usr/bin/env python3
"""shock_ride.py — ride basket price explosions, exit before crisis.

Uses DYNAMIC baskets from macro_sector_shock (GICS sectors + sub-industries
+ factor_groups) — not a fixed research ticker list.

Rule (per basket, monthly, no lookahead):
  ENTER  when 12m basket mom > entry_thresh (default 0.40) AND 3m mom > 0
  EXIT   when 3m mom <= 0
  position shifts 1 month after signals

Outputs:
  shock_ride.csv — basket, basket_kind, label, n_members, n_trades,
                   in_market_share, buy_hold_return, ride_return, excess,
                   max_dd_ride, max_dd_buyhold
Usage: python shock_ride.py [--save] [--entry 0.40]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_sector_shock import _build_baskets, _monthly_returns, _price_universe

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "shock_ride.csv"


def run(entry_thresh: float = 0.40, save: bool = True):
    have = _price_universe()
    baskets = _build_baskets(have)
    rows = []
    print(f"=== shock ride (entry: 12m mom > {entry_thresh:.0%}, exit: 3m mom <= 0) ===")
    print(f"  dynamic baskets: {len(baskets)}")
    for bid, cfg in sorted(baskets.items()):
        m = _monthly_returns(cfg["tickers"])
        if m.empty or len(m) < 24:
            continue
        cum = (1 + m).cumprod()
        mom12 = cum / cum.shift(12) - 1
        mom3 = cum / cum.shift(3) - 1
        pos = ((mom12 > entry_thresh) & (mom3 > 0)).astype(int)
        pos = pos.shift(1).fillna(0)

        strat = (pos * m).dropna()
        n_trades = int((pos.diff().fillna(0).abs() > 0).sum() // 2)
        in_share = float(pos.mean())
        ride = float(strat.sum())
        bh = float(m.dropna().sum())

        def max_dd(r):
            c = (1 + r).cumprod()
            return float((c / c.cummax() - 1).min())

        rows.append({
            "basket": bid,
            "basket_kind": cfg["kind"],
            "label": cfg["label"],
            "n_members": len(cfg["tickers"]),
            "n_trades": n_trades,
            "in_market_share": round(in_share, 3),
            "buy_hold_return": round(bh, 4),
            "ride_return": round(ride, 4),
            "excess": round(ride - bh, 4),
            "max_dd_ride": round(max_dd(strat), 4),
            "max_dd_buyhold": round(max_dd(m.dropna()), 4),
        })

    out = pd.DataFrame(rows).sort_values("excess", ascending=False)
    wins = int((out["excess"] > 0).sum()) if len(out) else 0
    print(f"\nBaskets where ride beats buy-hold: {wins}/{len(out)}")
    if len(out):
        print(f"Mean excess: {out['excess'].mean():+.1%} | "
              f"mean maxDD ride {out['max_dd_ride'].mean():.1%} vs BH {out['max_dd_buyhold'].mean():.1%}")
        print("\nTop 10 by excess:")
        for _, r in out.head(10).iterrows():
            print(f"  {r['basket'][:36]:36s} excess {r['excess']:+.1%}  "
                  f"ride {r['ride_return']:+.1%} BH {r['buy_hold_return']:+.1%}  n={int(r['n_members'])}")
    if save:
        out.to_csv(OUT, index=False)
        print(f"\nWrote {OUT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=0.40)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(entry_thresh=args.entry, save=True)

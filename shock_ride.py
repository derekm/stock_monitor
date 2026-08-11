#!/usr/bin/env python3
"""shock_ride.py — ride basket/ticker price explosions, exit before crisis.

Uses DYNAMIC baskets from macro_sector_shock (GICS sectors + sub-industries
+ factor_groups) AND a per-ticker ride rule over the full price universe.

Rule (per basket or per ticker, monthly, no lookahead):
  ENTER  when 12m mom > entry_thresh (default 0.40) AND 3m mom > 0
  EXIT   when 3m mom <= 0
  position shifts 1 month after signals

Outputs:
  shock_ride.csv — basket ride stats (dynamic baskets)
  shock_ride_tickers.csv — per-ticker ride stats + CURRENT position:
      ticker, name, sector, n_trades, in_market_share, buy_hold_return,
      ride_return, excess, max_dd_ride, max_dd_buyhold, mom1, mom3, mom12,
      ride_long (current), recommendation, interpretation
Usage: python shock_ride.py [--save] [--entry 0.40]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_sector_shock import _build_baskets, _load_price_matrix, _monthly_returns, _price_universe

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "shock_ride.parquet"
OUT_TICKERS = DATA_DIR / "shock_ride_tickers.parquet"
MIN_TICKER_HISTORY = 36  # months of price history required for a ticker ride
MAX_TICKERS = 600        # cap for the per-ticker pass (universe is ~583)


def _ride_stats(m: pd.Series, entry_thresh: float) -> dict:
    cum = (1 + m).cumprod()
    mom12 = cum / cum.shift(12) - 1
    mom3 = cum / cum.shift(3) - 1
    pos = ((mom12 > entry_thresh) & (mom3 > 0)).astype(int)
    pos = pos.shift(1).fillna(0)
    strat = (pos * m).dropna()

    def max_dd(r):
        c = (1 + r).cumprod()
        return float((c / c.cummax() - 1).min())

    mom1 = cum / cum.shift(1) - 1
    return {
        "n_trades": int((pos.diff().fillna(0).abs() > 0).sum() // 2),
        "in_market_share": float(pos.mean()),
        "buy_hold_return": round(float(m.dropna().sum()), 4),
        "ride_return": round(float(strat.sum()), 4),
        "excess": round(float(strat.sum()) - float(m.dropna().sum()), 4),
        "max_dd_ride": round(max_dd(strat), 4),
        "max_dd_buyhold": round(max_dd(m.dropna()), 4),
        "mom1": round(float(mom1.iloc[-1]), 4) if len(mom1) else np.nan,
        "mom3": round(float(mom3.iloc[-1]), 4) if len(mom3) else np.nan,
        "mom12": round(float(mom12.iloc[-1]), 4) if len(mom12) else np.nan,
        "ride_long": int(pos.iloc[-1]) if len(pos) else 0,
        "as_of": m.index[-1].strftime("%Y-%m-%d") if len(m) else "",
    }


def run(entry_thresh: float = 0.40, save: bool = True):
    have = _price_universe()
    baskets = _build_baskets(have)
    rows = []
    print(f"=== shock ride — dynamic baskets (entry: 12m mom > {entry_thresh:.0%}, exit: 3m mom <= 0) ===")
    print(f"  dynamic baskets: {len(baskets)}")
    for bid, cfg in sorted(baskets.items()):
        m = _monthly_returns(cfg["tickers"])
        if m.empty or len(m) < 24:
            continue
        st = _ride_stats(m, entry_thresh)
        rows.append({
            "basket": bid,
            "basket_kind": cfg["kind"],
            "label": cfg["label"],
            "n_members": len(cfg["tickers"]),
            **st,
        })

    out = pd.DataFrame(rows).sort_values("excess", ascending=False)
    wins = int((out["excess"] > 0).sum()) if len(out) else 0
    print(f"\nBaskets where ride beats buy-hold: {wins}/{len(out)}")
    if len(out):
        print(f"Mean excess: {out['excess'].mean():+.1%} | "
              f"mean maxDD ride {out['max_dd_ride'].mean():.1%} vs BH {out['max_dd_buyhold'].mean():.1%}")
    if save:
        out.to_parquet(OUT)
        print(f"Wrote {OUT}")

    # ── per-ticker ride pass ──
    print(f"\n=== per-ticker ride (universe {len(have)} tickers, min {MIN_TICKER_HISTORY}mo history) ===")
    w = _load_price_matrix()
    meta = None
    try:
        ms = pd.read_parquet(DATA_DIR / "monitored_stocks.parquet")
        meta = dict(zip(ms["ticker"].astype(str).str.upper(), ms["name"]))
        sec = dict(zip(ms["ticker"].astype(str).str.upper(), ms["sector"]))
    except Exception:
        pass

    trows = []
    tickers = sorted(t for t in have if t in w.columns)
    tickers = tickers[:MAX_TICKERS]
    for t in tickers:
        s = w[t].dropna()
        if len(s) < MIN_TICKER_HISTORY * 21:
            continue
        m = np.log(s / s.shift(1))
        m = m.replace([np.inf, -np.inf], np.nan).dropna()
        m = m.resample("ME").sum().dropna()
        if len(m) < MIN_TICKER_HISTORY:
            continue
        st = _ride_stats(m, entry_thresh)
        # recommendation (same honest logic as ride_now.py)
        hot = st["mom12"] > 0.40
        if st["ride_long"] and hot and st["mom3"] > 0:
            rec, interp = "BUY", (
                f"explosion still accelerating (12m {st['mom12']:+.0%}, "
                f"3m {st['mom3']:+.0%}, 1m {st['mom1']:+.0%})."
            )
        elif st["ride_long"] and hot:
            rec, interp = "STAND DOWN", (
                f"momentum says long (12m {st['mom12']:+.0%}, 3m {st['mom3']:+.0%}, "
                f"1m {st['mom1']:+.0%}) — 1m rolling over. Tighten stop to 3m rollover."
            )
        elif st["mom12"] > 0.40 and st["mom3"] <= 0:
            rec, interp = "AVOID", (
                f"exploded (12m {st['mom12']:+.0%}) but 3m {st['mom3']:+.0%} "
                f"(1m {st['mom1']:+.0%}) — rolled over; ride exited."
            )
        elif st["mom12"] > 0.40:
            rec, interp = "WATCH", (
                f"12m {st['mom12']:+.0%} — above threshold but 3m {st['mom3']:+.0%} "
                f"not yet positive; waiting for entry."
            )
        else:
            rec, interp = "FLAT", f"12m {st['mom12']:+.0%} / 3m {st['mom3']:+.0%} — no signal."
        trows.append({
            "ticker": t,
            "name": (meta or {}).get(t, ""),
            "sector": (sec or {}).get(t, ""),
            **st,
            "recommendation": rec,
            "interpretation": interp,
        })

    tout = pd.DataFrame(trows)
    order = {"BUY": 0, "STAND DOWN": 1, "AVOID": 2, "WATCH": 3, "FLAT": 4}
    tout["_o"] = tout["recommendation"].map(order)
    tout = tout.sort_values(["_o", "mom12"], ascending=[True, False]).drop(columns="_o")
    wins_t = int((tout["excess"] > 0).sum()) if len(tout) else 0
    print(f"\nTickers where ride beats buy-hold: {wins_t}/{len(tout)}")
    print(f"Mean excess: {tout['excess'].mean():+.1%} | "
          f"mean maxDD ride {tout['max_dd_ride'].mean():.1%} vs BH {tout['max_dd_buyhold'].mean():.1%}")
    print("\nRecommendations:", tout["recommendation"].value_counts().to_dict())
    print("\nTop 10 by excess:")
    for _, r in tout.head(10).iterrows():
        print(f"  {r['ticker']:6s} excess {r['excess']:+.1%}  ride {r['ride_return']:+.1%} BH {r['buy_hold_return']:+.1%}")
    if save:
        tout.to_parquet(OUT_TICKERS)
        print(f"\nWrote {OUT_TICKERS}")
    return out, tout


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=0.40)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(entry_thresh=args.entry, save=True)

#!/usr/bin/env python3
"""ride_now.py — CURRENT ride-rule state + recommendation per dynamic basket.

Answers "what is the ride indicator saying RIGHT NOW" with the honest
tension the momentum rule and the stress regime create. Read this before
sizing any basket exposure.

For every dynamic basket (GICS sector / sub-industry / factor group):

  ride_long = (12m mom > 0.40) AND (3m mom > 0)        [no lookahead]
  mom1 / mom3 / mom12 = 1/3/12-month basket momentum
  shock_zone = latest macro_sector_shock zone
  p_stress   = latest subindustry_regime stress posterior

Recommendation logic (table-driven, honest — the tension is surfaced, not
hidden):

  ride_long AND zone in (shock, elevated) AND p_stress < 0.8:
      BUY      "explosion still accelerating, regime calm"
  ride_long AND p_stress >= 0.8:
      STAND DOWN  "momentum says long, stress regime maxed — size small,
                   stop = 3m rollover"   (the contradiction is the signal)
  NOT ride_long AND zone in (shock, elevated):
      AVOID    "explosion already rolled over — buying the top"
  otherwise:
      FLAT     "no signal"

Outputs:
  ride_now.csv — basket, basket_kind, label, n_members, date, mom1, mom3,
                 mom12, ride_long, shock_zone, p_stress, regime,
                 recommendation, interpretation
Usage: python ride_now.py [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_sector_shock import _build_baskets, _monthly_returns, _price_universe

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "ride_now.csv"


def main(save: bool = True):
    have = _price_universe()
    baskets = _build_baskets(have)
    print(f"=== ride NOW · {len(baskets)} dynamic baskets ===")

    # latest shock zone per basket
    sec = pd.read_csv(DATA_DIR / "macro_sector_shock.csv")
    sec = sec.sort_values("date").groupby("basket", as_index=False).tail(1)
    zone_map = dict(zip(sec["basket"], sec["shock_zone"]))

    # latest regime per basket
    sub = pd.read_csv(DATA_DIR / "subindustry_regime.csv")
    sub = sub.sort_values("date").groupby("basket", as_index=False).tail(1)
    stress_map = dict(zip(sub["basket"], sub["p_stress"]))
    regime_map = dict(zip(sub["basket"], sub["regime"]))

    rows = []
    for bid, cfg in sorted(baskets.items()):
        m = _monthly_returns(cfg["tickers"])
        if m.empty or len(m) < 14:
            continue
        cum = (1 + m).cumprod()
        mom12 = float((cum / cum.shift(12) - 1).iloc[-1])
        mom3 = float((cum / cum.shift(3) - 1).iloc[-1])
        mom1 = float((cum / cum.shift(1) - 1).iloc[-1])
        ride_long = bool(mom12 > 0.40 and mom3 > 0)
        zone = zone_map.get(bid, "no_data")
        p_stress = float(stress_map.get(bid, np.nan))
        regime = regime_map.get(bid, "n/a")

        hot = zone in ("shock", "elevated")
        if ride_long and hot and (pd.isna(p_stress) or p_stress < 0.80):
            rec, interp = "BUY", (
                f"{cfg['label']} explosion still accelerating "
                f"(12m {mom12:+.0%}, 3m {mom3:+.0%}, 1m {mom1:+.0%}); "
                f"regime {regime} (p_stress {p_stress:.2f}) — cleanest ride-long now."
            )
        elif ride_long and (pd.isna(p_stress) or p_stress >= 0.80):
            rec, interp = "STAND DOWN", (
                f"momentum says long (12m {mom12:+.0%}, 3m {mom3:+.0%}, "
                f"1m {mom1:+.0%}) but regime is {regime} "
                f"(p_stress {p_stress:.2f}) — stress maxed. Size small, "
                f"stop = 3m rollover. The contradiction IS the signal."
            )
        elif not ride_long and hot:
            rec, interp = "AVOID", (
                f"{cfg['label']} exploded (12m {mom12:+.0%}) but 3m momentum "
                f"already {mom3:+.0%} (1m {mom1:+.0%}) — rolled over. "
                f"Ride rule exited; buying here is buying the top."
            )
        else:
            rec, interp = "FLAT", (
                f"12m {mom12:+.0%} / 3m {mom3:+.0%} — no ride entry."
            )

        rows.append({
            "basket": bid, "basket_kind": cfg["kind"], "label": cfg["label"],
            "n_members": len(cfg["tickers"]), "date": m.index[-1].strftime("%Y-%m-%d"),
            "mom1": round(mom1, 4), "mom3": round(mom3, 4), "mom12": round(mom12, 4),
            "ride_long": int(ride_long), "shock_zone": zone,
            "p_stress": round(p_stress, 4) if not pd.isna(p_stress) else np.nan,
            "regime": regime, "recommendation": rec, "interpretation": interp,
        })

    out = pd.DataFrame(rows)
    order = {"BUY": 0, "STAND DOWN": 1, "AVOID": 2, "FLAT": 3}
    out["_o"] = out["recommendation"].map(order)
    out = out.sort_values(["_o", "mom12"], ascending=[True, False]).drop(columns="_o")

    if save:
        out.to_csv(OUT, index=False)

    print("\n=== RECOMMENDATIONS (current) ===")
    print(out["recommendation"].value_counts().to_string())
    for _, r in out[out["recommendation"] != "FLAT"].head(15).iterrows():
        print(f"\n[{r['recommendation']}] {r['label']} ({r['basket']})")
        print(f"    mom 1m {r['mom1']:+.1%} 3m {r['mom3']:+.1%} 12m {r['mom12']:+.1%} | "
              f"zone {r['shock_zone']} | p_stress {r['p_stress']:.2f} {r['regime']}")
    if save:
        print(f"\nWrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

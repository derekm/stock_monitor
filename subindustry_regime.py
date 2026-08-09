#!/usr/bin/env python3
"""subindustry_regime.py — per-sub-industry correlation & crisis regimes,
plus an early-collapse warning that fires BEFORE the shock_ride momentum
exit bleeds.

Why it exists: shock_ride.py exits on 3m momentum rollover — a LAGGING
signal (the collapse is typically weeks old by then). The market-wide HMM
(hmm_regime_detection.py) is aggregate; sub-industries collapse
independently. This script runs the SAME 3-state HMM recipe per
sub-industry basket — features = basket vol21 + intra-basket avg pairwise
correlation — so each subsector gets its own stress posterior (the
"correlation & crisis regime per sub industry" ask), and measures whether
that subsector stress leads the ride rule's momentum exit.

Method (per basket, from hmm_regime_detection — same code, no fork):
  build_features(rets) -> vol21, avg_corr (21d rolling pairwise corr
  within the basket), mkt_ret
  fit_hmm(...) 3 states -> label_states: the high-vol/high-corr state is
  the subsector crisis regime (same labeling rule as the market HMM).
  p_stress = posterior prob of the crisis state.

Early-collapse test (the "before the ride rule takes losses" ask):
  At every shock_ride EXIT (3m mom rollover), what did the subsector
  p_stress read 10/20/30 days BEFORE the exit? If the subsector stress
  flips > 0.8 ahead of the momentum exit, the stress posterior is the
  leading signal and can replace/augment the ride exit.

Outputs:
  subindustry_regime.csv — per basket: date, vol21, avg_corr, p_stress,
                           regime, ride_pos (1 when ride rule is long)
  subindustry_regime_lead.csv — per basket: n_exits, exits with stress
                                already > 0.8 at -10/-20/-30d, mean lead
Usage: python subindustry_regime.py [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_regime_detection import build_features, fit_hmm, label_states
from macro_sector_shock import SECTORS, _monthly_returns

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "subindustry_regime.csv"
OUT_LEAD = DATA_DIR / "subindustry_regime_lead.csv"

# only the focused subsector baskets (sub_*) + the flagship fertilizer
BASKETS = {k: v for k, v in SECTORS.items() if k.startswith("sub_")}


def basket_daily_rets(cfg: dict) -> pd.DataFrame:
    """Daily (not monthly) equal-weight basket returns for the HMM
    features (vol21/corr are daily-frequency signals)."""
    import pandas as pd
    members = list(cfg.get("tickers") or [])
    try:
        sp = pd.read_parquet(DATA_DIR / "sp500_constituents.parquet")
        if cfg.get("gics"):
            members += sp.loc[sp["gics_sector"] == cfg["gics"], "ticker"].astype(str).str.upper().tolist()
        if cfg.get("subindustry"):
            members += sp.loc[sp["gics_sub_industry"] == cfg["subindustry"], "ticker"].astype(str).str.upper().tolist()
        members = list(dict.fromkeys(members))
    except Exception:
        pass
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    avail = [t for t in members if t in w.columns and w[t].notna().sum() > 500]
    if len(avail) < 2:
        return pd.DataFrame()
    rets = np.log(w[avail] / w[avail].shift(1))
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def ride_positions(cfg: dict) -> pd.Series:
    """Monthly ride-rule position (1 long / 0 flat) — same rule as
    shock_ride.py, monthly resample, 1-month shift."""
    m = _monthly_returns(tickers=cfg.get("tickers"), gics=cfg.get("gics"),
                         subindustry=cfg.get("subindustry"))
    if m.empty:
        return pd.Series(dtype=float)
    cum = (1 + m).cumprod()
    mom12 = cum / cum.shift(12) - 1
    mom3 = cum / cum.shift(3) - 1
    pos = ((mom12 > 0.40) & (mom3 > 0)).astype(int)
    return pos.shift(1).fillna(0)


def main(save: bool = True):
    rows, lead_rows = [], []
    print("=== sub-industry regimes (HMM vol21 + intra-basket corr) ===")
    for name, cfg in BASKETS.items():
        rets = basket_daily_rets(cfg)
        if rets.shape[1] < 2:
            print(f"  {name}: <2 members, skipped")
            continue
        feat = build_features(rets, corr_window=21)
        if len(feat) < 300:
            print(f"  {name}: short history ({len(feat)}), skipped")
            continue
        try:
            _, states, post, _, _ = fit_hmm(feat, n_states=3)
        except Exception as e:
            print(f"  {name}: HMM fail {str(e)[:50]}, skipped")
            continue
        labels, _ = label_states(feat, states)
        stress_state = [s for s, l in labels.items() if "stress" in l or "high" in l]
        stress_state = stress_state[0] if stress_state else labels[max(labels, key=lambda s: labels[s].count("high"))]
        p_stress = post[:, stress_state]

        d = pd.DataFrame({
            "date": feat.index, "vol21": feat["vol21"].values,
            "avg_corr": feat["avg_corr"].values,
            "p_stress": p_stress, "state": states,
        })
        d["regime"] = [labels[s] for s in states]
        # ride position at each date (monthly pos carried forward to daily)
        rp = ride_positions(cfg)
        rp_daily = rp.asof(d["date"]) if not rp.empty else pd.Series(0, index=d.index)
        d["ride_pos"] = rp_daily.to_numpy()
        d["basket"] = name
        rows.append(d)

        # lead analysis: at each ride EXIT (1 -> 0), stress 10/20/30d before
        exits = d[d["ride_pos"].diff() == -1]
        n_exits = len(exits)
        pre = {}
        for lag in (10, 20, 30):
            n_lead = 0
            for _, ex in exits.iterrows():
                i = d.index.get_loc(ex.name)
                if i - lag >= 0:
                    if d.iloc[i - lag]["p_stress"] > 0.8:
                        n_lead += 1
            pre[f"lead_{lag}d"] = n_lead
        lead_rows.append({"basket": name, "n_exits": n_exits, **pre})

        last = d.iloc[-1]
        print(f"  {name:22s} stress {last['p_stress']:.2f} "
              f"({last['regime']}) | vol {last['vol21']:.2f} corr {last['avg_corr']:.2f} "
              f"| exits {n_exits} lead10/20/30 {pre['lead_10d']}/{pre['lead_20d']}/{pre['lead_30d']}")

    if not rows:
        print("nothing computed")
        return
    out = pd.concat(rows, ignore_index=True)
    out = out[["basket", "date", "vol21", "avg_corr", "p_stress", "regime", "ride_pos"]]
    lead = pd.DataFrame(lead_rows)

    if save:
        out.to_csv(OUT, index=False)
        lead.to_csv(OUT_LEAD, index=False)

    # aggregate lead stats
    tot_exits = int(lead["n_exits"].sum())
    for lag in ("lead_10d", "lead_20d", "lead_30d"):
        tot = int(lead[lag].sum())
        pct = tot / tot_exits if tot_exits else 0
        print(f"\n  stress>0.8 BEFORE ride exit: {lag}: {tot}/{tot_exits} ({pct:.0%})")
    if save:
        print(f"\nWrote {OUT}\nWrote {OUT_LEAD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

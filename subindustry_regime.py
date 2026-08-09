#!/usr/bin/env python3
"""subindustry_regime.py — per-basket correlation & crisis regimes (DYNAMIC).

Runs the market HMM recipe (vol21 + intra-basket avg corr → 3-state stress)
on every dynamic basket from macro_sector_shock (GICS sub-industries +
sectors + factor_groups). Not a fixed research list.

Also measures whether p_stress > 0.8 LEADS the shock_ride momentum exit
(honest result historically: mostly coincident, not leading).

Outputs:
  subindustry_regime.csv — basket, basket_kind, label, date, vol21, avg_corr,
                           p_stress, regime, ride_pos
  subindustry_regime_lead.csv — basket, n_exits, lead_10d/20d/30d
Usage: python subindustry_regime.py [--save] [--max-baskets N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hmm_regime_detection import build_features, fit_hmm, label_states
from macro_sector_shock import _build_baskets, _monthly_returns, _price_universe

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "subindustry_regime.csv"
OUT_LEAD = DATA_DIR / "subindustry_regime_lead.csv"


def basket_daily_rets(tickers: list[str]) -> pd.DataFrame:
    """Daily equal-weight member returns (columns = tickers) for HMM features."""
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["ticker"].isin(tickers)]
    if p.empty:
        return pd.DataFrame()
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    avail = [t for t in w.columns if w[t].notna().sum() > 500]
    if len(avail) < 2:
        return pd.DataFrame()
    rets = np.log(w[avail] / w[avail].shift(1))
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def ride_positions(tickers: list[str], entry_thresh: float = 0.40) -> pd.Series:
    m = _monthly_returns(tickers)
    if m.empty:
        return pd.Series(dtype=float)
    cum = (1 + m).cumprod()
    mom12 = cum / cum.shift(12) - 1
    mom3 = cum / cum.shift(3) - 1
    pos = ((mom12 > entry_thresh) & (mom3 > 0)).astype(int)
    return pos.shift(1).fillna(0)


def main(save: bool = True, max_baskets: int | None = None):
    have = _price_universe()
    baskets = _build_baskets(have)
    # Prefer thinner baskets for regime (sub-industries + factor groups); still
    # include GICS sectors so the dashboard has full coverage.
    items = sorted(baskets.items())
    if max_baskets:
        items = items[:max_baskets]
    print(f"=== basket regimes (HMM vol21 + intra-basket corr) · {len(items)} baskets ===")

    rows, lead_rows = [], []
    for bid, cfg in items:
        tickers = cfg["tickers"]
        rets = basket_daily_rets(tickers)
        if rets.shape[1] < 2:
            continue
        feat = build_features(rets, corr_window=21)
        if len(feat) < 300:
            continue
        try:
            _, states, post, _, _ = fit_hmm(feat, n_states=3)
        except Exception as e:
            print(f"  {bid}: HMM fail {str(e)[:50]}")
            continue
        labels, _ = label_states(feat, states)
        stress_state = [s for s, l in labels.items() if "stress" in l or "high" in l]
        if not stress_state:
            # highest-vol state
            stress_state = [max(labels, key=lambda s: 0 if "low" in labels[s] else 1)]
        stress_state = stress_state[0]
        p_stress = post[:, stress_state]

        d = pd.DataFrame({
            "date": feat.index,
            "vol21": feat["vol21"].values,
            "avg_corr": feat["avg_corr"].values,
            "p_stress": p_stress,
            "state": states,
        })
        d["regime"] = [labels[s] for s in states]
        rp = ride_positions(tickers)
        if not rp.empty:
            # carry monthly position onto daily dates
            rp_daily = rp.reindex(d["date"]).ffill()
            # monthly index may not align — asof via merge_asof style
            if rp_daily.isna().all():
                tmp = pd.DataFrame({"date": d["date"]})
                rpdf = rp.rename("ride_pos").reset_index()
                rpdf.columns = ["date", "ride_pos"]
                rpdf["date"] = pd.to_datetime(rpdf["date"])
                tmp = pd.merge_asof(tmp.sort_values("date"), rpdf.sort_values("date"), on="date")
                d["ride_pos"] = tmp["ride_pos"].fillna(0).to_numpy()
            else:
                d["ride_pos"] = rp_daily.fillna(0).to_numpy()
        else:
            d["ride_pos"] = 0
        d["basket"] = bid
        d["basket_kind"] = cfg["kind"]
        d["label"] = cfg["label"]
        d["n_members"] = len(tickers)
        rows.append(d)

        exits = d[d["ride_pos"].diff() == -1]
        n_exits = len(exits)
        pre = {}
        for lag in (10, 20, 30):
            n_lead = 0
            for idx in exits.index:
                i = d.index.get_loc(idx)
                if isinstance(i, slice):
                    continue
                if i - lag >= 0 and d.iloc[i - lag]["p_stress"] > 0.8:
                    n_lead += 1
            pre[f"lead_{lag}d"] = n_lead
        lead_rows.append({
            "basket": bid, "basket_kind": cfg["kind"], "label": cfg["label"],
            "n_members": len(tickers), "n_exits": n_exits, **pre,
        })
        last = d.iloc[-1]
        print(f"  {bid[:34]:34s} stress {last['p_stress']:.2f} ({last['regime']}) "
              f"vol {last['vol21']:.2f} corr {last['avg_corr']:.2f} n={len(tickers)}")

    if not rows:
        print("nothing computed")
        return
    out = pd.concat(rows, ignore_index=True)
    out = out[["basket", "basket_kind", "label", "n_members", "date",
               "vol21", "avg_corr", "p_stress", "regime", "ride_pos"]]
    lead = pd.DataFrame(lead_rows)

    if save:
        out.to_csv(OUT, index=False)
        lead.to_csv(OUT_LEAD, index=False)

    tot_exits = int(lead["n_exits"].sum()) if len(lead) else 0
    for lag in ("lead_10d", "lead_20d", "lead_30d"):
        tot = int(lead[lag].sum()) if len(lead) else 0
        pct = tot / tot_exits if tot_exits else 0
        print(f"  stress>0.8 BEFORE ride exit {lag}: {tot}/{tot_exits} ({pct:.0%})")
    if save:
        print(f"\nWrote {OUT} ({len(out)} rows)\nWrote {OUT_LEAD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--max-baskets", type=int, default=None)
    args = ap.parse_args()
    main(save=True, max_baskets=args.max_baskets)

#!/usr/bin/env python3
"""barbell_check.py — convexity / barbell structure (the Taleb layer).

Why: a portfolio of mid-risk, mid-return assets is the "Christmas tree" —
exposed to every tail, rewarded for none. Taleb's barbell: ~90% in ultra-safe
(no downside) + ~10% in highly convex bets (capped downside, unbounded upside),
with nothing in the middle. This script checks the ACTUAL structure:

1. Exposure to volatility: regress portfolio daily returns on VIX-style
   volatility changes (realized vol of the portfolio). Negative beta to vol
   = short-vol = fragile (harvests premium, blown up in spikes).
2. Exposure to vol-of-vol (second-order): regress on |dvol| — the real
   fragility signal per Taleb.
3. Convexity decomposition: split the portfolio into safe / middle / convex
   buckets by volatility and gap share; report the barbell score
   (weight in safe + weight in convex, minus weight in middle).
4. Hedge sizing: cost of a protective put ladder (from options_skew ATM IV)
   and the recommended permanent convexity allocation (fragility-based).

Outputs:
  barbell_check.csv      per-bucket weights, barbell score, vol beta,
                         vol-of-vol beta, hedge cost estimate, convexity
                         allocation recommendation
Reads: daily_prices.parquet, gap_risk.csv, options_skew.csv, holdings.
Usage: python barbell_check.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252


def main():
    # portfolio: equal-weight daily returns (aligned on real dates)
    cols = ["date", "ticker", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    d = d.sort_values(["ticker", "date"])
    port = None
    n_tick = 0
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change().dropna()
        if len(r) < 500:
            continue
        s = r.copy()
        s.index = pd.to_datetime(g["date"].iloc[1:1 + len(s)])
        port = s if port is None else port.add(s, fill_value=0)
        n_tick += 1
    if port is None or n_tick == 0:
        print("no data")
        return
    pr = (port / n_tick).dropna()

    # realized vol (21d) and its change -> vol beta and vol-of-vol beta
    rv = pr.rolling(21).std() * np.sqrt(TRADING_DAYS)
    dv = rv.diff()
    df = pd.DataFrame({"ret": pr, "rv": rv, "dv": dv}).dropna()
    r = df["ret"].to_numpy()
    dvol = df["dv"].to_numpy()
    vol_beta = float(np.polyfit(dvol, r, 1)[0]) if len(df) > 60 else np.nan
    # vol-of-vol exposure: |dv| (magnitude of vol changes) vs return
    vov_beta = float(np.polyfit(np.abs(dvol), r, 1)[0]) if len(df) > 60 else np.nan

    # per-name buckets: safe (low vol + low gap), middle, convex (high vol or high gap)
    try:
        g = pd.read_csv(DATA_DIR / "gap_risk.csv")
    except Exception:
        g = pd.DataFrame(columns=["ticker", "gap_share_of_var", "ret_sd"])
    vol_ser = {}
    for t, gd in d.groupby("ticker"):
        r_ = gd["close"].pct_change().dropna()
        if len(r_) > 500:
            vol_ser[t] = float(r_.std() * np.sqrt(TRADING_DAYS))
    vol_df = pd.DataFrame({"ticker": list(vol_ser.keys()), "ann_vol": list(vol_ser.values())})
    vol_df = vol_df.merge(g[["ticker", "gap_share_of_var"]], on="ticker", how="left")
    vol_df["gap_share"] = vol_df["gap_share_of_var"].fillna(0.5)
    # barbell buckets (relative to THIS universe: it has no <15% vol names)
    # safe = bottom-quartile vol + low gap share; convex = top-quintile vol OR
    # extreme gap share; middle = everything else
    vq = vol_df["ann_vol"].quantile(0.25)
    vh = vol_df["ann_vol"].quantile(0.80)
    gh = vol_df["gap_share"].quantile(0.90)
    vol_df["bucket"] = np.where(
        (vol_df["ann_vol"] <= vq) & (vol_df["gap_share"] <= gh), "safe",
        np.where((vol_df["ann_vol"] >= vh) | (vol_df["gap_share"] > 1.2), "convex", "middle"))
    counts = vol_df["bucket"].value_counts()
    n = len(vol_df)
    w_safe = counts.get("safe", 0) / n
    w_mid = counts.get("middle", 0) / n
    w_conv = counts.get("convex", 0) / n
    barbell = w_safe + w_conv - w_mid

    # hedge cost: average ATM IV from options_skew as put cost proxy
    try:
        sk = pd.read_csv(DATA_DIR / "options_skew.csv")
        atm = float(sk["atm_iv"].mean()) if len(sk) else np.nan
        put_cost = atm * 0.3  # ~30% of IV for an ATM 3m put
    except Exception:
        atm, put_cost = np.nan, np.nan
    # recommended convexity allocation: scale with portfolio fragility
    try:
        fs = pd.read_csv(DATA_DIR / "fragility_screen.csv")
        avg_frag = float(fs["fragility_pctile"].mean())
    except Exception:
        avg_frag = 0.5
    alloc = round(min(0.15, max(0.02, 0.05 + 0.10 * avg_frag)), 3)

    rows = [
        {"metric": "n_names", "value": n},
        {"metric": "weight_safe", "value": round(w_safe, 3)},
        {"metric": "weight_middle", "value": round(w_mid, 3)},
        {"metric": "weight_convex", "value": round(w_conv, 3)},
        {"metric": "barbell_score", "value": round(barbell, 3)},
        {"metric": "vol_beta", "value": round(vol_beta, 4) if np.isfinite(vol_beta) else None},
        {"metric": "vol_of_vol_beta", "value": round(vov_beta, 4) if np.isfinite(vov_beta) else None},
        {"metric": "avg_atm_iv", "value": round(atm, 4) if np.isfinite(atm) else None},
        {"metric": "put_ladder_cost_ann", "value": round(put_cost, 4) if np.isfinite(put_cost) else None},
        {"metric": "recommended_convexity_alloc", "value": alloc},
    ]
    pd.DataFrame(rows).to_csv(DATA_DIR / "barbell_check.csv", index=False)

    print(f"barbell_check.csv written | n={n}")
    for r_ in rows:
        print(f"  {r_['metric']:28s} {r_['value']}")
    print("\nInterpretation:")
    print(f"  barbell_score {'> 0.5: near-barbell' if barbell > 0.5 else '< 0.5: Christmas tree (middle-heavy)'} "
          f"(actual {barbell:.2f})")
    if np.isfinite(vol_beta):
        print(f"  vol_beta {vol_beta:+.3f} -> {'short-vol / fragile (gains in calm, bleeds in spikes)' if vol_beta < 0 else 'long-vol / convex'}")
    if np.isfinite(vov_beta):
        print(f"  vol-of-vol beta {vov_beta:+.3f} -> {'fragile to vol-of-vol' if vov_beta < 0 else 'benefits from vol-of-vol'}")


if __name__ == "__main__":
    main()

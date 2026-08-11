#!/usr/bin/env python3
"""
binding_constraints_analysis.py — Impact of dual-pass binding constraints.

For each of the six legs:
  - count how many names fail that leg (alone and jointly)
  - measure "shadow dual" set if that leg is removed
  - risk metrics of base dual vs leave-one-out baskets
  - distance-to-threshold for near-misses on binding legs (ROIC, MCA, …)

Usage:
  python binding_constraints_analysis.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
OUT = DATA_DIR / "binding_constraints_impact.parquet"
OUT_NEAR = DATA_DIR / "binding_near_miss_detail.parquet"
OUT_RISK = DATA_DIR / "binding_basket_risk.parquet"

BASE = dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9.0, pb_max=1.5, mca_max=0.5)


def latest_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def pass_mask(df, roe_min, roic_min, de_max, ev_max, pb_max, mca_max):
    return (
        (df["roe"] >= roe_min) & (df["roic"] >= roic_min) & (df["debt_to_equity"] <= de_max)
        & (df["ev_ebitda"] <= ev_max) & (df["pb_ratio"] <= pb_max) & (df["mktcap_to_assets"] <= mca_max)
    )


def basket_risk(tickers, rets) -> dict:
    cols = [t for t in tickers if t in rets.columns]
    if len(cols) < 1:
        return dict(n=0, port_vol=np.nan, max_dd=np.nan, avg_name_vol=np.nan)
    r = rets[cols].mean(axis=1)
    cum = r.cumsum()
    wealth = np.exp(cum)
    dd = wealth / wealth.cummax() - 1
    return dict(
        n=len(cols),
        port_vol=float(r.std() * np.sqrt(252)),
        max_dd=float(dd.min()),
        avg_name_vol=float(rets[cols].std().mean() * np.sqrt(252)),
    )


def run(save: bool = True):
    fund = latest_fund()
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")

    base_m = pass_mask(fund, **BASE)
    base_tickers = fund.loc[base_m, "ticker"].tolist()
    print(f"Base dual-pass ({len(base_tickers)}): {base_tickers}")

    # Per-leg fail counts (among names with non-null metrics)
    legs = {
        "roe": lambda d: d["roe"] >= BASE["roe_min"],
        "roic": lambda d: d["roic"] >= BASE["roic_min"],
        "de": lambda d: d["debt_to_equity"] <= BASE["de_max"],
        "ev": lambda d: d["ev_ebitda"] <= BASE["ev_max"],
        "pb": lambda d: d["pb_ratio"] <= BASE["pb_max"],
        "mca": lambda d: d["mktcap_to_assets"] <= BASE["mca_max"],
    }

    rows = []
    for name, fn in legs.items():
        ok = fn(fund)
        n_fail = int((~ok).sum())
        n_pass_leg = int(ok.sum())
        # dual if this leg ignored
        others = [fn2 for n2, fn2 in legs.items() if n2 != name]
        shadow = fund[pd.concat([fn2(fund) for fn2 in others], axis=1).all(axis=1)]
        shadow_t = shadow["ticker"].tolist()
        new = sorted(set(shadow_t) - set(base_tickers))
        risk = basket_risk(shadow_t, rets)
        base_risk = basket_risk(base_tickers, rets)
        rows.append({
            "leg": name,
            "n_fail_leg": n_fail,
            "n_pass_leg": n_pass_leg,
            "shadow_dual_n": len(shadow_t),
            "new_if_dropped": ",".join(new),
            "n_new": len(new),
            "shadow_port_vol": risk["port_vol"],
            "shadow_max_dd": risk["max_dd"],
            "base_port_vol": base_risk["port_vol"],
            "vol_delta": (risk["port_vol"] - base_risk["port_vol"]) if pd.notna(risk["port_vol"]) else np.nan,
        })
        vol_s = f"{risk['port_vol']:.3f}" if risk['port_vol']==risk['port_vol'] else "nan"
        print(f"  drop {name:4s}: shadow n={len(shadow_t):2d}  new={new[:8]}  vol={vol_s}")

    impact = pd.DataFrame(rows).sort_values("n_new", ascending=False)

    # Distance to threshold for non-dual names on each leg
    near_rows = []
    for _, x in fund.iterrows():
        if x["ticker"] in base_tickers:
            continue
        gaps = {
            "roe_gap": float(x["roe"] - BASE["roe_min"]) if pd.notna(x["roe"]) else np.nan,
            "roic_gap": float(x["roic"] - BASE["roic_min"]) if pd.notna(x["roic"]) else np.nan,
            "de_gap": float(BASE["de_max"] - x["debt_to_equity"]) if pd.notna(x["debt_to_equity"]) else np.nan,
            "ev_gap": float(BASE["ev_max"] - x["ev_ebitda"]) if pd.notna(x["ev_ebitda"]) else np.nan,
            "pb_gap": float(BASE["pb_max"] - x["pb_ratio"]) if pd.notna(x["pb_ratio"]) else np.nan,
            "mca_gap": float(BASE["mca_max"] - x["mktcap_to_assets"]) if pd.notna(x["mktcap_to_assets"]) else np.nan,
        }
        # failed = negative gap for min-rules, negative for max-rules already signed as (threshold - value) or (value - min)
        n_fail = sum([
            gaps["roe_gap"] < 0 if pd.notna(gaps["roe_gap"]) else True,
            gaps["roic_gap"] < 0 if pd.notna(gaps["roic_gap"]) else True,
            gaps["de_gap"] < 0 if pd.notna(gaps["de_gap"]) else True,
            gaps["ev_gap"] < 0 if pd.notna(gaps["ev_gap"]) else True,
            gaps["pb_gap"] < 0 if pd.notna(gaps["pb_gap"]) else True,
            gaps["mca_gap"] < 0 if pd.notna(gaps["mca_gap"]) else True,
        ])
        near_rows.append({"ticker": x["ticker"], "n_fail": n_fail, **gaps,
                          "roe": x["roe"], "roic": x["roic"], "mktcap_to_assets": x["mktcap_to_assets"],
                          "pb_ratio": x["pb_ratio"], "ev_ebitda": x["ev_ebitda"]})
    near = pd.DataFrame(near_rows)
    # binding focus: fail only 1 leg
    one_leg = near[near.n_fail == 1].copy()
    print("\n=== Single-leg failures (true binding near-misses) ===")
    if len(one_leg):
        # identify which leg
        def which(row):
            fails = []
            if row.roe_gap < 0: fails.append("roe")
            if row.roic_gap < 0: fails.append("roic")
            if row.de_gap < 0: fails.append("de")
            if row.ev_gap < 0: fails.append("ev")
            if row.pb_gap < 0: fails.append("pb")
            if row.mca_gap < 0: fails.append("mca")
            return ",".join(fails)
        one_leg["failed_leg"] = one_leg.apply(which, axis=1)
        print(one_leg[["ticker", "failed_leg", "roe", "roic", "mktcap_to_assets", "pb_ratio", "ev_ebitda"]].to_string(index=False))
        print("\nBinding leg counts (single-fail only):")
        print(one_leg["failed_leg"].value_counts().to_string())
    else:
        print("  (none)")

    # two-leg fails
    two = near[near.n_fail == 2]
    print(f"\nTwo-leg failures: {len(two)} names (near-dual zone)")

    risk_rows = [{"basket": "base_dual", **basket_risk(base_tickers, rets), "tickers": ",".join(base_tickers)}]
    for name in legs:
        others = [fn2 for n2, fn2 in legs.items() if n2 != name]
        shadow = fund[pd.concat([fn2(fund) for fn2 in others], axis=1).all(axis=1)]
        risk_rows.append({"basket": f"drop_{name}", **basket_risk(shadow["ticker"].tolist(), rets),
                          "tickers": ",".join(shadow["ticker"].tolist()[:15])})
    risk_df = pd.DataFrame(risk_rows)
    print("\n=== Basket risk by dropped constraint ===")
    print(risk_df.to_string(index=False))

    if save:
        impact.to_parquet(OUT)
        near.sort_values("n_fail").to_parquet(OUT_NEAR)
        risk_df.to_parquet(OUT_RISK)
        print(f"\nWrote {OUT}\nWrote {OUT_NEAR}\nWrote {OUT_RISK}")
    return impact, near, risk_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
stress_dual_pass.py — Stress-test dual-pass inclusion criteria.

Varies ROE/ROIC/D/E/EV/P/B/MCA thresholds and reports how many names pass.
Also runs one-leg relaxation sensitivity.

Usage:
  python stress_dual_pass.py
  python stress_dual_pass.py --save
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "dual_pass_stress.parquet"
OUT_SENS = DATA_DIR / "dual_pass_sensitivity.parquet"


def latest_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)



def risk_of_names(df, tickers, prices_path=DATA_DIR / "daily_prices.parquet"):
    """Portfolio EW risk metrics for a ticker list."""
    if not tickers:
        return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"))
    try:
        prices = pd.read_parquet(prices_path, columns=["date", "ticker", "close"])
        prices["date"] = pd.to_datetime(prices["date"])
        wide = prices[prices.ticker.isin(tickers)].pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        rets = np.log(wide / wide.shift(1)).dropna(how="all")
        if rets.empty:
            return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"))
        ew = rets.mean(axis=1)
        port_vol = float(ew.std() * np.sqrt(252))
        cum = ew.cumsum()
        max_dd = float((np.exp(cum) / np.exp(cum).cummax() - 1).min())
        name_vol = rets.std() * np.sqrt(252)
        mkt = rets.mean(axis=1)
        betas = []
        for c in rets.columns:
            cov = np.cov(rets[c].dropna().align(mkt, join="inner")[0], rets[c].dropna().align(mkt, join="inner")[1])
            # simpler:
            aligned = pd.concat([rets[c], mkt], axis=1, keys=["a","m"]).dropna()
            if len(aligned) > 20 and aligned["m"].var() > 0:
                betas.append(float(aligned.cov().iloc[0,1] / aligned["m"].var()))
        return dict(
            port_vol=port_vol,
            port_max_dd=max_dd,
            avg_beta=float(np.mean(betas)) if betas else float("nan"),
            avg_name_vol=float(name_vol.mean()) if len(name_vol) else float("nan"),
            n_names=len(tickers),
        )
    except Exception as e:
        return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"), error=str(e))

def count_pass(df, roe_min, roic_min, de_max, ev_max, pb_max, mca_max):
    m = (
        (df["roe"] >= roe_min) & (df["roic"] >= roic_min) & (df["debt_to_equity"] <= de_max)
        & (df["ev_ebitda"] <= ev_max) & (df["pb_ratio"] <= pb_max) & (df["mktcap_to_assets"] <= mca_max)
    )
    return int(m.sum()), df.loc[m, "ticker"].tolist()


def run(save: bool = True):
    df = latest_fund()
    base = dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9.0, pb_max=1.5, mca_max=0.5)
    n0, t0 = count_pass(df, **base)
    print(f"Base dual-pass: {n0} names → {t0}")

    # Grid stress
    grid = {
        "roe_min": [0.10, 0.12, 0.15, 0.18, 0.20],
        "roic_min": [0.10, 0.12, 0.15, 0.18],
        "de_max": [0.5, 1.0, 1.5, 2.0],
        "ev_max": [7.0, 9.0, 12.0, 15.0],
        "pb_max": [1.0, 1.5, 2.0, 3.0],
        "mca_max": [0.3, 0.5, 0.8, 1.2],
    }
    rows = []
    # one-parameter-at-a-time from base
    for param, values in grid.items():
        for val in values:
            kw = dict(base)
            kw[param] = val
            n, tickers = count_pass(df, **kw)
            risk = risk_of_names(df, tickers)
            rows.append({
                "mode": "one_at_a_time",
                "param": param,
                "value": str(val),
                "n_pass": n,
                "delta_vs_base": n - n0,
                "tickers": ",".join(tickers[:20]),
                **{k: risk.get(k) for k in ("port_vol","port_max_dd","avg_beta","avg_name_vol")},
            })
    # joint relaxed / tight scenarios
    scenarios = [
        ("tight", dict(roe_min=0.18, roic_min=0.18, de_max=0.5, ev_max=7, pb_max=1.0, mca_max=0.3)),
        ("base", base),
        ("relaxed_quality", dict(roe_min=0.12, roic_min=0.12, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("relaxed_value", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=12, pb_max=2.0, mca_max=0.8)),
        ("relaxed_both", dict(roe_min=0.12, roic_min=0.12, de_max=1.5, ev_max=12, pb_max=2.0, mca_max=0.8)),
        ("buffett_fair", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=15, pb_max=3.0, mca_max=1.5)),
    ]
    for name, kw in scenarios:
        n, tickers = count_pass(df, **kw)
        risk = risk_of_names(df, tickers)
        rows.append({
            "mode": "scenario",
            "param": name,
            "value": str(kw),
            "n_pass": n,
            "delta_vs_base": n - n0,
            "tickers": ",".join(tickers[:25]),
            **{k: risk.get(k) for k in ("port_vol","port_max_dd","avg_beta","avg_name_vol")},
        })
        print(f"  scenario {name:16s}  n={n:3d}  ({n-n0:+d})  vol={risk.get('port_vol') and risk['port_vol']*100:.1f}%  {tickers[:6]}")

    # leave-one-leg-out sensitivity
    sens = []
    legs = [
        ("drop_roe", dict(roe_min=-9, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_roic", dict(roe_min=0.15, roic_min=-9, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_de", dict(roe_min=0.15, roic_min=0.15, de_max=99, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_ev", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=99, pb_max=1.5, mca_max=0.5)),
        ("drop_pb", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=99, mca_max=0.5)),
        ("drop_mca", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=99)),
    ]
    print("\nLeave-one-leg-out:")
    for name, kw in legs:
        n, tickers = count_pass(df, **kw)
        new = sorted(set(tickers) - set(t0))
        sens.append({"dropped_leg": name, "n_pass": n, "delta": n - n0, "new_tickers": ",".join(new[:15])})
        print(f"  {name:12s} n={n:3d} (+{n-n0}) new={new[:10]}")

    out = pd.DataFrame(rows)
    sens_df = pd.DataFrame(sens)
    if save:
        out.to_parquet(OUT)
        sens_df.to_parquet(OUT_SENS)
        print(f"Wrote {OUT}\nWrote {OUT_SENS}")
    return out, sens_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

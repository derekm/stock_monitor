#!/usr/bin/env python3
"""
tail_risk_hedging.py — Explore tail-risk hedging overlays for the defensive book.

Hedges evaluated (overlay on equal-weight defensive index):
  1. cash_buffer      — hold 10–20% cash (zero return)
  2. low_vol_tilt     — 30% min-vol names / defensive ETFs
  3. put_proxy        — short market factor on worst days (vol-scaled short mkt)
  4. gold_proxy       — synthetic defensive alternative (low beta sleeve)
  5. dual_quality     — dual-pass + quality only core
  6. tail_hedge_combo — cash 10% + low_vol 20% + defensive 70%

Metrics: ann vol, max DD, CVaR 5%, crisis-period return, Sharpe.

Usage:
  python tail_risk_hedging.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from threshold_logic import select_regime_from_hmm_file
except ImportError:
    select_regime_from_hmm_file = None


DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "tail_risk_hedge_performance.parquet"
OUT_CRISIS = DATA_DIR / "tail_risk_hedge_crisis.parquet"


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 10:
        return {}
    cum = r.cumsum()
    wealth = np.exp(cum)
    dd = wealth / wealth.cummax() - 1
    var5 = float(r.quantile(0.05))
    cvar5 = float(r[r <= var5].mean()) if (r <= var5).any() else var5
    return {
        "ann_ret": float(r.mean() * 252),
        "ann_vol": float(r.std() * np.sqrt(252)),
        "sharpe": float(r.mean() * 252 / (r.std() * np.sqrt(252))) if r.std() > 0 else np.nan,
        "max_dd": float(dd.min()),
        "cvar_5pct": cvar5,
        "var_5pct": var5,
    }


def run(save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS)
    fund = pd.read_parquet(FUND)
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"])
    fund = fund.sort_values("as_of_date").groupby("ticker").tail(1).set_index("ticker")

    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)

    def ew(cols):
        cols = [c for c in cols if c in rets.columns]
        return rets[cols].mean(axis=1) if cols else pd.Series(0.0, index=rets.index)

    defensive = stocks.loc[stocks.get("defensive_value_index") == True, "ticker"].tolist()
    etfs = stocks.loc[stocks.get("value_sleeve") == "defensive_etf", "ticker"].tolist() if "value_sleeve" in stocks.columns else []
    vols = rets.iloc[-63:].std() * np.sqrt(252)
    low_vol = vols.nsmallest(max(8, len(vols) // 4)).index.tolist()
    dual = fund[(fund.roe >= 0.15) & (fund.roic >= 0.15) & (fund.debt_to_equity <= 1) &
                (fund.ev_ebitda <= 9) & (fund.pb_ratio <= 1.5) & (fund.mktcap_to_assets <= 0.5)].index.tolist()
    quality = fund[(fund.roe >= 0.15) & (fund.roic >= 0.15) & (fund.debt_to_equity <= 1)].index.tolist()

    base = ew(defensive)
    # put proxy: on days when mkt < -1.5%, subtract 0.5 * mkt move (partial hedge)
    put = base.copy()
    hedge_days = mkt < -0.015
    put = base - 0.4 * mkt.where(hedge_days, 0.0)

    strategies = {
        "defensive_ew": base,
        "cash_10": 0.9 * base,
        "cash_20": 0.8 * base,
        "low_vol_tilt": 0.7 * base + 0.3 * ew(low_vol),
        "dividend_etf_30": 0.7 * base + 0.3 * ew(etfs),
        "put_proxy": put,
        "dual_quality": 0.5 * ew(dual) + 0.5 * ew(quality),
        "tail_combo": 0.10 * 0 + 0.20 * ew(low_vol) + 0.20 * ew(etfs) + 0.50 * base,
    }

    # crisis window: top vol days
    vol21 = mkt.rolling(21).std() * np.sqrt(252)
    crisis = vol21 >= vol21.quantile(0.8)

    
    # Regime-aware recommended overlay (consumes HMM via threshold_logic)
    regime = "normal"
    if select_regime_from_hmm_file is not None:
        try:
            regime = select_regime_from_hmm_file(soft_min=0.7)
        except Exception:
            regime = "normal"
    if regime == "high_vol_stress":
        strategies["regime_recommended"] = strategies["tail_combo"]
    elif regime == "low_vol":
        strategies["regime_recommended"] = strategies["defensive_ew"]
    elif regime == "uncertain":
        strategies["regime_recommended"] = strategies["cash_10"]
    else:
        strategies["regime_recommended"] = 0.85 * strategies["defensive_ew"] + 0.15 * strategies["low_vol_tilt"]
    print(f"HMM regime={regime} → regime_recommended overlay selected")

    rows = []
    crisis_rows = []
    print("=== Tail-risk hedge performance ===")
    for name, series in strategies.items():
        s = stats(series)
        s["strategy"] = name
        # crisis only
        cr = stats(series.loc[crisis.reindex(series.index).fillna(False)])
        s["crisis_ann_ret"] = cr.get("ann_ret")
        s["crisis_vol"] = cr.get("ann_vol")
        s["crisis_cvar"] = cr.get("cvar_5pct")
        rows.append(s)
        crisis_rows.append({"strategy": name, **{f"crisis_{k}": v for k, v in cr.items()}})
        print(f"{name:16s} vol={s.get('ann_vol',float('nan'))*100:5.1f}%  maxDD={s.get('max_dd',float('nan'))*100:6.1f}%  "
              f"CVaR5={s.get('cvar_5pct',float('nan'))*100:5.2f}%  crisisRet={s.get('crisis_ann_ret',float('nan'))*100:6.1f}%")

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(crisis_rows)

    # Honest OOS: regime_recommended vs defensive_ew on the last 2 years only
    # (full-history stats mix regimes; this is what a live book saw recently).
    from cv_utils import oos_stats_vs_baseline
    rec = strategies.get("regime_recommended")
    base = strategies.get("defensive_ew")
    if rec is not None and base is not None and len(rec) > 504 and len(base) > 504:
        oos = oos_stats_vs_baseline(rec.tail(504), base.tail(504))
        oos["strategy"] = "regime_recommended_vs_defensive_ew_OOS_2y"
        df = pd.concat([df, pd.DataFrame([oos])], ignore_index=True)
        print("\n=== OOS 2y: regime_recommended vs defensive_ew ===")
        for k, v in oos.items():
            if k != "strategy":
                print(f"  {k}: {v}")

    if save:
        df.to_parquet(OUT)
        cdf.to_parquet(OUT_CRISIS)
        print(f"Wrote {OUT}\nWrote {OUT_CRISIS}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

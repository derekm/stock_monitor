#!/usr/bin/env python3
"""
factor_rotation_defense.py — Defensive factor rotation strategies.

Factors (long-only sleeves, equal-weight):
  quality   — Buffett pass names
  value     — trifecta pass names
  low_vol   — bottom tercile 63d vol
  dividend  — defensive ETFs (SCHD/VIG/XLP/XLU/…) + high earnings_stability
  dual      — INCLUDE_CORE dual-pass

Rotation signals (monthly):
  - Risk-on: prior 21d market vol below median → overweight quality/dual
  - Risk-off: vol above 80th pct or crisis flag → overweight low_vol + dividend ETFs
  - Value tilt when value-quality spread (value minus quality 63d return) is depressed

Usage:
  python factor_rotation_defense.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from threshold_logic import select_regime_from_hmm_file, thresholds_for_regime
except ImportError:
    select_regime_from_hmm_file = None
    thresholds_for_regime = None


DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
PREF = DATA_DIR / "preferred_metrics.csv"
OUT_W = DATA_DIR / "factor_rotation_weights.csv"
OUT_PERF = DATA_DIR / "factor_rotation_performance.csv"
OUT_SLEEVE = DATA_DIR / "factor_sleeve_returns.csv"


def latest_fund():
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def build_sleeves(rets: pd.DataFrame) -> dict[str, list[str]]:
    stocks = pd.read_parquet(STOCKS)
    fund = latest_fund().set_index("ticker")
    vols = rets.iloc[-63:].std() * np.sqrt(252)
    low_vol = vols.nsmallest(max(5, len(vols)//3)).index.tolist()

    buffett = fund[(fund.roe >= 0.15) & (fund.roic >= 0.15) & (fund.debt_to_equity <= 1.0)].index.tolist()
    trifecta = fund[(fund.ev_ebitda <= 9) & (fund.pb_ratio <= 1.5) & (fund.mktcap_to_assets <= 0.5)].index.tolist()
    dual = list(set(buffett) & set(trifecta))

    etfs = stocks.loc[stocks.get("value_sleeve") == "defensive_etf", "ticker"].tolist() if "value_sleeve" in stocks.columns else []
    if not etfs:
        etfs = [t for t in ["SCHD", "VIG", "XLP", "XLU", "USMV", "SPLV", "VYM"] if t in rets.columns]

    def filt(xs):
        return [t for t in xs if t in rets.columns]

    return {
        "quality": filt(buffett),
        "value": filt(trifecta),
        "dual": filt(dual),
        "low_vol": filt(low_vol),
        "dividend": filt(etfs),
        "defensive_idx": filt(stocks.loc[stocks.get("defensive_value_index") == True, "ticker"].tolist()),
    }


def sleeve_return(rets: pd.DataFrame, members: list[str]) -> pd.Series:
    cols = [c for c in members if c in rets.columns]
    if not cols:
        return pd.Series(0.0, index=rets.index)
    return rets[cols].mean(axis=1)


def run(save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)

    sleeves = build_sleeves(rets)
    print("Sleeve sizes:", {k: len(v) for k, v in sleeves.items()})
    for k, v in sleeves.items():
        print(f"  {k}: {v[:8]}{'...' if len(v)>8 else ''}")

    sleeve_rets = {k: sleeve_return(rets, v) for k, v in sleeves.items()}
    sret = pd.DataFrame(sleeve_rets)

    # monthly rotation
    months = sret.index.to_period("M").unique()
    weights_rows = []
    port = []
    for i, m in enumerate(months):
        # signal from prior month-end
        end = (m.start_time - pd.Timedelta(days=1))
        hist = vol21.loc[:end].dropna()
        risk_off = False
        risk_on = False
        value_cheap = False
        # HMM / threshold_logic regime (if available) overrides pure vol flags
        regime = "normal"
        if select_regime_from_hmm_file is not None:
            try:
                regime = select_regime_from_hmm_file(soft_min=0.7)
            except Exception:
                regime = "normal"
        if len(hist) < 22:
            w = {"quality": 0.2, "value": 0.2, "low_vol": 0.2, "dividend": 0.2, "dual": 0.1, "defensive_idx": 0.1}
        else:
            v = float(hist.iloc[-1])
            v_med = float(hist.iloc[-126:].median()) if len(hist) >= 60 else float(hist.median())
            v_p80 = float(hist.iloc[-126:].quantile(0.8)) if len(hist) >= 60 else float(hist.quantile(0.8))
            risk_off = v >= v_p80 or regime == "high_vol_stress"
            risk_on = (v <= v_med and regime == "low_vol") or (regime == "low_vol")
            if regime == "uncertain":
                risk_off = False
                risk_on = False  # neutral when HMM is unsure
            # value-quality spread
            q = sret["quality"].loc[:end].iloc[-63:].sum() if len(sret.loc[:end]) >= 63 else 0
            val = sret["value"].loc[:end].iloc[-63:].sum() if len(sret.loc[:end]) >= 63 else 0
            value_cheap = val < q  # value lagged quality
            if risk_off:
                w = {"low_vol": 0.30, "dividend": 0.30, "defensive_idx": 0.15, "quality": 0.10, "value": 0.10, "dual": 0.05}
            elif risk_on and not value_cheap:
                w = {"quality": 0.25, "dual": 0.20, "defensive_idx": 0.15, "value": 0.15, "low_vol": 0.15, "dividend": 0.10}
            elif value_cheap:
                w = {"value": 0.30, "dual": 0.15, "quality": 0.15, "low_vol": 0.15, "dividend": 0.15, "defensive_idx": 0.10}
            else:
                w = {"quality": 0.20, "value": 0.20, "low_vol": 0.20, "dividend": 0.15, "dual": 0.10, "defensive_idx": 0.15}
        # days in month
        days = sret.index[sret.index.to_period("M") == m]
        for d in days:
            weights_rows.append({"date": d, **w, "regime": "risk_off" if risk_off else ("risk_on" if risk_on else "neutral"), "hmm_regime": regime})
            r = sum(w.get(k, 0) * float(sret.loc[d, k]) if d in sret.index and k in sret.columns else 0 for k in w)
            port.append({"date": d, "ret": r})

    wdf = pd.DataFrame(weights_rows)
    pdf = pd.DataFrame(port).set_index("date")["ret"]
    # EW defensive static benchmark
    static = sret.get("defensive_idx", sret.mean(axis=1))
    # performance
    def stats(r):
        r = r.dropna()
        if len(r) < 5:
            return {}
        return {
            "ann_ret": float(r.mean() * 252),
            "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() * 252 / (r.std() * np.sqrt(252))) if r.std() > 0 else np.nan,
            "max_dd": float((np.exp(r.cumsum()) / np.exp(r.cumsum()).cummax() - 1).min()),
        }

    perf = []
    for name, series in list(sleeve_rets.items()) + [("rotation", pdf), ("static_defensive", static)]:
        s = stats(series)
        s["strategy"] = name
        perf.append(s)
    perf_df = pd.DataFrame(perf)
    print("\n=== Factor / rotation performance ===")
    print(perf_df.to_string(index=False))

    if save:
        wdf.to_csv(OUT_W, index=False)
        perf_df.to_csv(OUT_PERF, index=False)
        sret.reset_index().rename(columns={"index": "date"}).to_csv(OUT_SLEEVE, index=False)
        print(f"Wrote {OUT_W}\nWrote {OUT_PERF}\nWrote {OUT_SLEEVE}")
    return perf_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

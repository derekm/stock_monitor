#!/usr/bin/env python3
"""momentum_research_backtest.py — backtest the research momentum measures.

Tests each measure's predictive power across the full price universe and finds
reliable confidence thresholds. For each measure and each ticker, at each month
it records whether the signal is ON and the forward k-month return. Aggregates
to hit-rate, mean forward return, and (where a long-only position is implied)
Sharpe. Compares signal-on vs signal-off to find which measures/conditions
actually separate winners from losers.

Measures tested:
  TSMOM 3/6/12 (JFE 2012), JT 6-1 (JT 1993), STMOM 1m (RFS 2022),
  GW-52w high (George-Hwang 2004), and the young-gate.

Usage: python momentum_research_backtest.py [--tickers N] [--window 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_research import (
    tsmom_signal, stmom_1m, gw52_high, jt_momentum, young_gate,
    ENTRY_THRESH, MIN_POST_IPO_MONTHS, VOL_CAP,
)

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "momentum_research_backtest.parquet"


def _monthly_log_returns(wide: pd.DataFrame) -> pd.DataFrame:
    r = np.log(wide / wide.shift(1))
    return r.replace([np.inf, -np.inf], np.nan).resample("ME").sum()


def forward_cum(m: pd.Series, horizon: int) -> pd.Series:
    """Forward horizon-month log return ending at t+horizon, indexed at t."""
    cum = m.cumsum()
    fwd = cum.shift(-horizon) - cum
    return fwd


def run(tickers_cap: int | None = None) -> pd.DataFrame:
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_cap:
        tickers = tickers[:tickers_cap]

    m_all = _monthly_log_returns(w)
    # cross-sectional ADV (liquidity) proxy: mean |monthly return| (turnover proxy)
    adv_proxy = m_all.abs().mean(axis=1)

    rows = []
    for t in tickers:
        m = m_all[t].replace([np.inf, -np.inf], np.nan).dropna()
        if len(m) < 8:
            continue
        # signals (all long-only, proper 0/1 via >0 boolean, not int-truncation)
        ts3 = (tsmom_signal(m, 3, vol_scaled=False) > 0).astype(float)
        ts6 = (tsmom_signal(m, 6, vol_scaled=False) > 0).astype(float)
        ts12 = (tsmom_signal(m, 12, vol_scaled=False) > 0).astype(float)
        jt6 = m.cumsum().diff(6).gt(0).astype(float)          # JT 6-mo formation
        sm1 = (m > 0).astype(float)                           # STMOM 1-mo
        cum = m.cumsum()
        hi12 = cum.rolling(12, min_periods=1).max()
        gw = (cum / hi12).fillna(0) >= 0.90                    # GW near-high
        ann_vol = m.rolling(12).std() * np.sqrt(12)
        vol_ok = ann_vol <= VOL_CAP

        for i in range(len(m) - 1):
            if i < 6:
                continue
            # forward returns at 3/6/12 mo
            for h in (3, 6, 12):
                if i + h >= len(m):
                    continue
                fwd = float((cum.iloc[i + h] - cum.iloc[i]))
                base = {
                    "ticker": t,
                    "date": m.index[i],
                    "horizon": h,
                    "fwd_log_ret": round(fwd, 4),
                }
                # feature signals (current, not lagged — this is an IC-style test)
                base["tsmom_3"] = int(ts3.iloc[i])
                base["tsmom_6"] = int(ts6.iloc[i])
                base["tsmom_12"] = int(ts12.iloc[i])
                base["jt_6"] = int(jt6.iloc[i])
                base["stmom_1"] = int(sm1.iloc[i])
                base["gw_high"] = int(gw.iloc[i])
                base["vol_ok"] = int(vol_ok.iloc[i])
                base["age_mo"] = int(i)
                rows.append(base)

    df = pd.DataFrame(rows)
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hit-rate + mean forward return for each signal, on vs off."""
    feats = ["tsmom_3", "tsmom_6", "tsmom_12", "jt_6", "stmom_1", "gw_high", "vol_ok"]
    out = []
    for f in feats:
        for h in (3, 6, 12):
            sub = df[df["horizon"] == h]
            on = sub[sub[f] == 1]["fwd_log_ret"]
            off = sub[sub[f] == 0]["fwd_log_ret"]
            if len(on) < 50 or len(off) < 50:
                continue
            out.append({
                "feature": f,
                "horizon": h,
                "n_on": len(on),
                "hit_rate_on": round((on > 0).mean(), 3),
                "mean_on": round(on.mean(), 4),
                "mean_off": round(off.mean(), 4),
                "spread": round(on.mean() - off.mean(), 4),
                "annualized_spread": round((on.mean() - off.mean()) * 12 / h, 3),
            })
    r = pd.DataFrame(out)
    r = r.sort_values("annualized_spread", ascending=False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    ap.add_argument("--window", type=int, default=60, help="min months per ticker")
    args = ap.parse_args()

    print("Building feature matrix (signal-on vs forward return)...")
    df = run(tickers_cap=args.tickers)
    print(f"  rows: {len(df)} | tickers: {df['ticker'].nunique()}")
    if df.empty:
        print("no data")
        return

    r = report(df)
    pd.set_option("display.width", 200)
    print("\n=== Measure predictive power (signal-on vs signal-off forward log return) ===")
    print(r.to_string(index=False) if not r.empty else "no features met min sample")

    r.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

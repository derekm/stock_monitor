#!/usr/bin/env python3
"""momentum_research_confidence.py — test signal AGREEMENT as a confidence measure.

The single-measure backtest (momentum_research_backtest.py) shows each of TSMOM
3/6/12, JT-6, STMOM-1 and GW-high has positive predictive power (hit rate
~0.60-0.67). This script tests the core hypothesis: does requiring MULTIPLE
independent momentum measures to AGREE raise reliability (hit rate, spread)?
This is the confidence measure we can trust for entries.

It rebuilds the same feature matrix and then buckets by the NUMBER of signals on
(1..6) and by which combinations fire, computing hit-rate / mean-forward / spread
per bucket. A monotone rise in reliability with signal-count = a usable confidence
measure.

Usage: python momentum_research_confidence.py [--tickers N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_research import tsmom_signal, ENTRY_THRESH, VOL_CAP

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "momentum_research_confidence.parquet"


def _monthly_log_returns(wide: pd.DataFrame) -> pd.DataFrame:
    r = np.log(wide / wide.shift(1))
    return r.replace([np.inf, -np.inf], np.nan).resample("ME").sum()


def build_matrix(tickers_cap: int | None = None) -> pd.DataFrame:
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_cap:
        tickers = tickers[:tickers_cap]
    m_all = _monthly_log_returns(w)

    rows = []
    for t in tickers:
        m = m_all[t].replace([np.inf, -np.inf], np.nan).dropna()
        if len(m) < 13:
            continue
        ts3 = (tsmom_signal(m, 3, vol_scaled=False) > 0).astype(float)
        ts6 = (tsmom_signal(m, 6, vol_scaled=False) > 0).astype(float)
        ts12 = (tsmom_signal(m, 12, vol_scaled=False) > 0).astype(float)
        jt6 = m.cumsum().diff(6).gt(0).astype(float)
        sm1 = (m > 0).astype(float)
        cum = m.cumsum()
        hi12 = cum.rolling(12, min_periods=1).max()
        gw = (cum / hi12).fillna(0) >= 0.90
        for i in range(12, len(m) - 1):
            for h in (3, 6, 12):
                if i + h >= len(m):
                    continue
                rows.append({
                    "ticker": t,
                    "date": m.index[i],
                    "horizon": h,
                    "fwd_log_ret": float(cum.iloc[i + h] - cum.iloc[i]),
                    "tsmom_3": int(ts3.iloc[i]),
                    "tsmom_6": int(ts6.iloc[i]),
                    "tsmom_12": int(ts12.iloc[i]),
                    "jt_6": int(jt6.iloc[i]),
                    "stmom_1": int(sm1.iloc[i]),
                    "gw_high": int(gw.iloc[i]),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    args = ap.parse_args()

    print("Building feature matrix...")
    df = build_matrix(args.tickers)
    print(f"  rows: {len(df)}")
    if df.empty:
        return

    feats = ["tsmom_3", "tsmom_6", "tsmom_12", "jt_6", "stmom_1", "gw_high"]
    df["n_signals"] = df[feats].sum(axis=1)

    rows = []
    for h in (3, 6, 12):
        sub = df[df["horizon"] == h]
        for n in range(1, 7):
            b = sub[sub["n_signals"] == n]
            if len(b) < 100:
                continue
            base = sub[sub["n_signals"] == 0]
            base_mean = base["fwd_log_ret"].mean() if len(base) else np.nan
            rows.append({
                "horizon": h,
                "n_signals_on": n,
                "n": len(b),
                "hit_rate": round((b["fwd_log_ret"] > 0).mean(), 3),
                "mean_fwd": round(b["fwd_log_ret"].mean(), 4),
                "base_mean_0sig": round(base_mean, 4) if pd.notna(base_mean) else np.nan,
                "spread_vs_0": round(b["fwd_log_ret"].mean() - base_mean, 4) if pd.notna(base_mean) else np.nan,
                "annual_spread_vs_0": round((b["fwd_log_ret"].mean() - base_mean) * 12 / h, 3) if pd.notna(base_mean) else np.nan,
            })
    res = pd.DataFrame(rows)

    # also: the full-agreement combo (all 6 on) vs partial
    pd.set_option("display.width", 200)
    print("\n=== Confidence: reliability by number of agreeing momentum signals ===")
    print("(monotone rise in hit-rate/spread with signal-count = usable confidence)")
    for h in (3, 6, 12):
        print(f"\n--- horizon {h}mo ---")
        hb = res[res["horizon"] == h].sort_values("n_signals_on")
        print(hb.to_string(index=False) if not hb.empty else "  (no buckets)")

    res.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

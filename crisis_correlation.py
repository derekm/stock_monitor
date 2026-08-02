#!/usr/bin/env python3
"""
crisis_correlation.py — Correlation breakdown in stress / crisis regimes.

Defines crisis windows as:
  1) Top-quintile market vol days (realized 21d vol)
  2) Worst 5% market return days
  3) Drawdown episodes (market below -8% from peak)

Compares avg pairwise corr: calm vs crisis.

Usage:
  python crisis_correlation.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT = DATA_DIR / "crisis_correlation_summary.csv"
OUT_PAIR = DATA_DIR / "crisis_correlation_pairs.csv"
OUT_TS = DATA_DIR / "crisis_avg_corr_timeseries.csv"


def avg_pairwise(corr: pd.DataFrame) -> float:
    v = corr.values
    n = v.shape[0]
    if n < 2:
        return float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    return float(np.nanmean(v[mask]))


def run(save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS, columns=["ticker", "sector", "defensive_value_index", "growth_tech_index", "value_sleeve", "instrument_type"])

    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)

    # regimes
    vol_cut = vol21.quantile(0.8)
    crisis_vol = vol21 >= vol_cut
    crisis_ret = mkt <= mkt.quantile(0.05)
    peak = mkt.cumsum().cummax()
    dd = mkt.cumsum() - peak
    crisis_dd = dd <= dd.quantile(0.1)

    crisis = crisis_vol | crisis_ret | crisis_dd
    calm = ~crisis & vol21.notna()

    def corr_in(mask):
        idx = rets.index[mask.reindex(rets.index).fillna(False)]
        if len(idx) < 15:
            return None, 0
        return rets.loc[idx].corr(), len(idx)

    results = []
    pair_rows = []
    for name, mask in [("calm", calm), ("crisis_vol", crisis_vol), ("crisis_ret", crisis_ret),
                       ("crisis_dd", crisis_dd), ("crisis_any", crisis)]:
        c, n = corr_in(mask)
        if c is None:
            results.append({"regime": name, "n_days": n, "avg_pairwise_corr": np.nan})
            continue
        avg = avg_pairwise(c)
        results.append({"regime": name, "n_days": n, "avg_pairwise_corr": avg,
                        "median_pairwise": float(np.nanmedian(c.values[np.triu(np.ones(c.shape, bool), 1)]))})
        print(f"{name:12s} days={n:4d}  avg_corr={avg:.3f}")

    # sector-level crisis vs calm
    sector_map = stocks.set_index("ticker")["sector"].to_dict()
    for regime_name, mask in [("calm", calm), ("crisis_any", crisis)]:
        idx = rets.index[mask.reindex(rets.index).fillna(False)]
        if len(idx) < 15:
            continue
        block = rets.loc[idx]
        sret = {}
        for sec in sorted(set(sector_map.values())):
            cols = [t for t in block.columns if sector_map.get(t) == sec]
            if len(cols) >= 2:
                sret[sec] = block[cols].mean(axis=1)
        if len(sret) < 2:
            continue
        sc = pd.DataFrame(sret).corr()
        avg = avg_pairwise(sc)
        results.append({"regime": f"sector_{regime_name}", "n_days": len(idx), "avg_pairwise_corr": avg})
        print(f"sector_{regime_name:8s} avg_corr={avg:.3f}")

    # pair-level: largest correlation increase crisis vs calm
    c_calm, _ = corr_in(calm)
    c_cris, _ = corr_in(crisis)
    if c_calm is not None and c_cris is not None:
        common = c_calm.columns.intersection(c_cris.columns)
        for i, a in enumerate(common):
            for b in common[i+1:]:
                pair_rows.append({
                    "asset_a": a, "asset_b": b,
                    "corr_calm": float(c_calm.loc[a, b]),
                    "corr_crisis": float(c_cris.loc[a, b]),
                    "delta": float(c_cris.loc[a, b] - c_calm.loc[a, b]),
                })
        pairs = pd.DataFrame(pair_rows).sort_values("delta", ascending=False)
        print("\nLargest corr increases in crisis:")
        print(pairs.head(10).to_string(index=False))
        print("\nLargest corr decreases (diversifiers in stress):")
        print(pairs.tail(10).sort_values("delta").to_string(index=False))
        if save:
            pairs.to_csv(OUT_PAIR, index=False)

    # rolling avg pairwise timeseries tagged with crisis flag
    window = 21
    ts_rows = []
    for i in range(window, len(rets)):
        block = rets.iloc[i-window:i]
        avg = avg_pairwise(block.corr())
        dt = rets.index[i]
        ts_rows.append({
            "date": dt,
            "avg_pairwise_corr": avg,
            "mkt_vol21": float(vol21.loc[dt]) if dt in vol21.index and pd.notna(vol21.loc[dt]) else np.nan,
            "crisis": bool(crisis.loc[dt]) if dt in crisis.index else False,
        })
    ts = pd.DataFrame(ts_rows)

    summary = pd.DataFrame(results)
    if save:
        summary.to_csv(OUT, index=False)
        ts.to_csv(OUT_TS, index=False)
        print(f"\nWrote {OUT}\nWrote {OUT_TS}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
regime_correlation_breakdown.py — Correlation structure inside each HMM regime.

Computes:
  - Average / median pairwise asset correlation by regime
  - Sector EW correlation by regime
  - Top pairs that spike most in high_vol_stress vs low_vol
  - Diversification ratio proxy (1 / avg corr intensity)

Usage:
  python regime_correlation_breakdown.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
HMM = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "regime_corr_breakdown.parquet"
OUT_PAIRS = DATA_DIR / "regime_corr_pair_delta.parquet"
OUT_SEC = DATA_DIR / "regime_sector_corr.parquet"


def avg_pairwise(corr: pd.DataFrame) -> tuple[float, float]:
    v = corr.values
    n = v.shape[0]
    if n < 2:
        return np.nan, np.nan
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    vals = v[mask]
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def run(save: bool = True, max_assets: int = 60):
    hmm = pd.read_parquet(HMM)
    hmm["date"] = pd.to_datetime(hmm["date"])
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS, columns=["ticker", "sector"])
    sector_map = stocks.set_index("ticker")["sector"].to_dict()

    # liquid names
    counts = prices.groupby("ticker").size().sort_values(ascending=False)
    tickers = counts.index.tolist()[:max_assets]
    wide = (
        prices[prices.ticker.isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index().ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all")

    rows = []
    pair_store = {}
    sec_rows = []

    for regime, g in hmm.groupby("regime"):
        idx = rets.index.intersection(pd.to_datetime(g["date"]))
        if len(idx) < 15:
            continue
        block = rets.loc[idx]
        c = block.corr()
        avg, med = avg_pairwise(c)
        # diversification proxy
        div = 1.0 / avg if avg and avg > 0 else np.nan
        rows.append({
            "regime": regime, "n_days": len(idx), "n_assets": c.shape[0],
            "avg_pairwise_corr": avg, "median_pairwise_corr": med,
            "diversification_proxy": div,
            "mean_name_vol": float(block.std().mean() * np.sqrt(252)),
            "mkt_vol": float(block.mean(axis=1).std() * np.sqrt(252)),
        })
        print(f"{regime:16s} days={len(idx):3d}  avgρ={avg:.3f}  medρ={med:.3f}  mktσ={rows[-1]['mkt_vol']:.3f}")

        # store pairs
        cols = list(c.columns)
        pairs = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                pairs[(a, b)] = float(c.loc[a, b])
        pair_store[regime] = pairs

        # sector (skip unmapped: sorting NaN floats with strs raises TypeError)
        sret = {}
        for sec in sorted({s for s in sector_map.values() if isinstance(s, str) and s.strip()}):
            scols = [t for t in block.columns if sector_map.get(t) == sec]
            if len(scols) >= 2:
                sret[sec] = block[scols].mean(axis=1)
        if len(sret) >= 2:
            sc = pd.DataFrame(sret).corr()
            savg, smed = avg_pairwise(sc)
            sec_rows.append({"regime": regime, "avg_sector_corr": savg, "median_sector_corr": smed})
            for i, a in enumerate(sc.columns):
                for b in list(sc.columns)[i + 1 :]:
                    sec_rows.append({
                        "regime": regime, "sector_a": a, "sector_b": b, "corr": float(sc.loc[a, b]),
                    })

    summary = pd.DataFrame([r for r in rows])

    # pair deltas stress - calm
    deltas = []
    if "high_vol_stress" in pair_store and "low_vol" in pair_store:
        common = set(pair_store["high_vol_stress"]) & set(pair_store["low_vol"])
        for ab in common:
            d = pair_store["high_vol_stress"][ab] - pair_store["low_vol"][ab]
            deltas.append({
                "asset_a": ab[0], "asset_b": ab[1],
                "corr_low_vol": pair_store["low_vol"][ab],
                "corr_stress": pair_store["high_vol_stress"][ab],
                "delta": d,
            })
        ddf = pd.DataFrame(deltas).sort_values("delta", ascending=False)
        print("\nLargest corr increases in stress:")
        print(ddf.head(8).to_string(index=False))
        print("\nLargest corr decreases (stress diversifiers):")
        print(ddf.tail(5).sort_values("delta").to_string(index=False))
    else:
        ddf = pd.DataFrame()

    sec_df = pd.DataFrame(sec_rows)
    if save:
        summary.to_parquet(OUT)
        ddf.to_parquet(OUT_PAIRS)
        sec_df.to_parquet(OUT_SEC)
        print(f"\nWrote {OUT}\nWrote {OUT_PAIRS}\nWrote {OUT_SEC}")
    return summary, ddf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

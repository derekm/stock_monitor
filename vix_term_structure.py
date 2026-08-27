#!/usr/bin/env python3
"""VIX / vol term-structure exploration (offline realized-vol proxy)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
LIVE = DATA_DIR / "vix_term_structure_live.parquet"
OUT = DATA_DIR / "vix_term_structure.parquet"
OUT_SUM = DATA_DIR / "vix_term_structure_summary.parquet"

def synthetic_curve(mkt: pd.Series) -> pd.DataFrame:
    horizons = [5, 10, 21, 42, 63]
    cols = {f"rv_{h}d": mkt.rolling(h).std() * np.sqrt(252) * 100 for h in horizons}
    df = pd.DataFrame(cols, index=mkt.index).dropna()
    df["VIX_proxy"] = df["rv_21d"]
    df["VIX3M_proxy"] = df["rv_63d"]
    df["VIX9D_proxy"] = df["rv_10d"]
    df["term_slope_21_63"] = df["VIX3M_proxy"] - df["VIX_proxy"]
    df["term_slope_10_21"] = df["VIX_proxy"] - df["VIX9D_proxy"]
    df["contango"] = df["term_slope_21_63"] > 0
    df["backwardation"] = df["term_slope_21_63"] < 0
    df["curve_convexity"] = (df["rv_10d"] + df["rv_63d"]) / 2 - df["rv_21d"]
    return df

def run(save: bool = True):
    if LIVE.exists():
        curve = pd.read_parquet(LIVE)
        curve["date"] = pd.to_datetime(curve["date"])
        source = "live"
    else:
        prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
        prices["date"] = pd.to_datetime(prices["date"])
        wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        mkt = np.log(wide / wide.shift(1)).mean(axis=1)
        curve = synthetic_curve(mkt).reset_index().rename(columns={"index": "date"})
        source = "synthetic_rv_proxy"
        print("No live VIX — using realized-vol term-structure proxy")
    curve["source"] = source
    print(curve[[c for c in curve.columns if "proxy" in c or "slope" in c]].describe().to_string())
    if "contango" in curve.columns:
        print(f"Contango {curve.contango.mean()*100:.1f}%  Backwardation {curve.backwardation.mean()*100:.1f}%")
        print(curve[["date","VIX9D_proxy","VIX_proxy","VIX3M_proxy","term_slope_21_63"]].tail().to_string(index=False))
    summary = pd.DataFrame([{
        "source": source,
        "pct_contango": float(curve["contango"].mean()) if "contango" in curve.columns else np.nan,
        "mean_slope_21_63": float(curve["term_slope_21_63"].mean()) if "term_slope_21_63" in curve.columns else np.nan,
        "last_vix_proxy": float(curve["VIX_proxy"].iloc[-1]) if "VIX_proxy" in curve.columns else np.nan,
    }])
    if save:
        curve.to_parquet(OUT)
        summary.to_parquet(OUT_SUM)
        print(f"Wrote {OUT}")
    return curve

if __name__ == "__main__":
    run(save=True)

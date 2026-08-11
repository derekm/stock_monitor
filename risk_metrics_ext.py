#!/usr/bin/env python3
"""risk_metrics_ext.py — Liquidity, concentration, factor-style risk (Polars + pandas).

Outputs:
  risk_metrics_ext.csv — per-ticker liquidity + simple factor scores
  portfolio_risk_summary.csv — portfolio concentration / liquidity / beta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"
HOLD = DATA_DIR / "portfolio_holdings.parquet"
PREF = DATA_DIR / "preferred_metrics.parquet"
OUT = DATA_DIR / "risk_metrics_ext.parquet"
OUT_PORT = DATA_DIR / "portfolio_risk_summary.parquet"


def dollar_volume(days: int = 21) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(str(PRICES))
        .with_columns(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("volume") > 0)
        .with_columns((pl.col("close") * pl.col("volume")).alias("dvol"))
        .sort(["ticker", "date"])
        .group_by("ticker")
        .agg([
            pl.col("dvol").tail(days).mean().alias("adv_dollar_21"),
            pl.col("volume").tail(days).mean().alias("adv_shares_21"),
            pl.col("close").last().alias("last_close"),
            pl.col("date").max().alias("as_of"),
        ])
    )
    return lf.collect()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    liq = dollar_volume().to_pandas()
    pref = pd.read_parquet(PREF) if PREF.exists() else pd.DataFrame()
    if len(pref):
        df = pref.merge(liq, on="ticker", how="left")
    else:
        df = liq
    # liquidity score 0-1 by rank of ADV
    if "adv_dollar_21" in df.columns:
        df["liquidity_score"] = df["adv_dollar_21"].rank(pct=True)
        df["illiquid"] = df["adv_dollar_21"] < df["adv_dollar_21"].quantile(0.2)
    # simple factor-style scores from existing metrics
    if "roe" in df.columns:
        df["factor_quality"] = df["roe"].rank(pct=True)
    if "ev_ebitda" in df.columns:
        df["factor_value"] = (1.0 / df["ev_ebitda"].replace(0, np.nan)).rank(pct=True)
    if "name_vol" in df.columns:
        df["factor_lowvol"] = (1.0 / df["name_vol"].replace(0, np.nan)).rank(pct=True)
    if "beta" in df.columns:
        df["factor_beta"] = df["beta"]

    # portfolio summary
    port = {"n_names": 0}
    if HOLD.exists():
        h = pd.read_parquet(HOLD)
        if "weight" in h.columns:
            w = h.set_index("ticker")["weight"].astype(float)
            if w.sum() > 2:
                w = w / 100.0
            w = w / w.sum() if w.sum() else w
            port = {
                "n_names": int(len(w)),
                "hhi": float((w ** 2).sum()),
                "top1_weight": float(w.max()) if len(w) else 0,
                "top3_weight": float(w.nlargest(min(3, len(w))).sum()) if len(w) else 0,
                "effective_n": float(1.0 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0,
            }
            merged = h.merge(liq, on="ticker", how="left")
            if "adv_dollar_21" in merged.columns and len(w):
                port["w_avg_adv_dollar_21"] = float(
                    (merged.set_index("ticker")["adv_dollar_21"].reindex(w.index).fillna(0) * w).sum()
                )
            if "beta" in df.columns:
                b = df.set_index("ticker")["beta"].reindex(w.index).astype(float)
                port["w_avg_beta"] = float((b.fillna(1.0) * w).sum())
    return df, pd.DataFrame([port])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    df, port = build()
    print(port.to_string(index=False))
    print(df[["ticker", "adv_dollar_21", "liquidity_score"]].dropna().head(8).to_string(index=False)
          if "liquidity_score" in df.columns else df.head())
    if args.save:
        df.to_parquet(OUT)
        port.to_parquet(OUT_PORT)
        print(f"Wrote {OUT.name}, {OUT_PORT.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gap_risk.py — overnight gap exposure (the Taleb layer).

Vectorized implementation using polars groupby.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-events", type=int, default=40)
    args = ap.parse_args()

    cols = ["date", "ticker", "open", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices/", columns=cols)
    d = d.sort_values(["ticker", "date"])

    # Vectorized gap calculation
    d["prev_close"] = d.groupby("ticker")["close"].shift(1)
    d["gap"] = d["open"] / d["prev_close"] - 1
    d["ret"] = d["close"] / d["prev_close"] - 1
    
    # Filter valid rows
    valid = d.dropna(subset=["gap", "ret"])
    
    # Groupby agg (vectorized)
    gap_stats = valid.groupby("ticker").agg(
        n_obs=("gap", "count"),
        mean_gap=("gap", "mean"),
        gap_sd=("gap", "std"),
        ret_sd=("ret", "std"),
        max_gap=("gap", lambda x: x.abs().max()),
        min_gap=("gap", "min"),
    ).reset_index()
    
    # Filter >= 200 obs
    gap_stats = gap_stats[gap_stats["n_obs"] >= 200]
    
    # Vectorized probability calcs
    gap_probs = valid[valid["ticker"].isin(gap_stats["ticker"])].groupby("ticker").agg(
        p3=("gap", lambda x: np.mean(np.abs(x) > 0.03)),
        p5=("gap", lambda x: np.mean(np.abs(x) > 0.05)),
    ).reset_index()
    
    # Merge
    df = gap_stats.merge(gap_probs, on="ticker")
    df["gap_share_of_var"] = (df["gap_sd"] ** 2 / df["ret_sd"] ** 2).where(df["ret_sd"] > 0)
    
    # Round
    df = df.round({
        "mean_gap": 5, "gap_sd": 5, "ret_sd": 5,
        "gap_share_of_var": 3, "p3": 5, "p5": 6,
        "max_gap": 4, "min_gap": 4
    })
    
    # Convert gap to pct
    df["max_gap_pct"] = (df["max_gap"] * 100).round(2)
    df["min_gap_pct"] = (df["min_gap"] * 100).round(2)
    
    df = df.sort_values("ticker")
    df.to_parquet(DATA_DIR / "gap_risk.parquet")

    # Events
    events = valid[valid["gap"].abs() > 0.05].copy()
    if len(events):
        events["gap_pct"] = (events["gap"] * 100).round(2)
        events["close_pct"] = (events["ret"] * 100).round(2)
        events = events[["ticker", "date", "gap_pct", "close_pct"]]
        events["date"] = events["date"].astype(str).str[:10]
        events = events.reindex(events["gap_pct"].abs().sort_values(ascending=False).index).head(args.top_events)
    events.to_parquet(DATA_DIR / "gap_events.parquet")

    print(f"gap_risk.csv: {len(df)} tickers")
    print(f"gap_events.csv: {len(events)} events")
    if len(df):
        risky = df.sort_values("gap_share_of_var", ascending=False).head(8)
        print("\nHighest gap share of variance — risk arrives overnight, backtests miss it:")
        print(risky[["ticker", "gap_share_of_var", "p3", "max_gap_pct"]].to_string(index=False))
        big = df.sort_values("p5", ascending=False).head(5)
        print("\nMost gap-prone names (P(|gap|>5%)):")
        print(big[["ticker", "p5", "max_gap_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
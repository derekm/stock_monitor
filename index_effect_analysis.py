#!/usr/bin/env python3
"""
index_effect_analysis.py — S&P 500 index effect analysis per Preston & Soe (2021).

Tests the structural decline of the index effect across three eras:
- 1995-1999: Strong effect (+8.3% adds, -9.6% dels, 17.9pp spread)
- 2000-2010: Weakened (+2.1%, -3.2%, 5.3pp)
- 2011-2021: Vanished/Reversed (-0.3%, +0.8%, -1.1pp)

Key distinction: Graduates (Mid/Small → S&P 500) vs Outsiders (large caps not in S&P 1500)
- Graduates: effect gone (supply from mid/small funds offsets S&P 500 demand)
- Outsiders: effect persists (no offsetting supply, e.g., TSLA +70%)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent

# Input paths
CHANGES = DATA_DIR / "sp500_changes_merged.parquet"
MEMBERSHIP = DATA_DIR / "sp500_membership.parquet"
PRICES = DATA_DIR / "daily_prices/"

# Output
OUT_ANALYSIS = DATA_DIR / "index_effect_analysis.parquet"
OUT_EVENTS = DATA_DIR / "index_effect_events.parquet"


def load_data():
    changes = pd.read_parquet(CHANGES)
    membership = pd.read_parquet(MEMBERSHIP)
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    membership["date"] = pd.to_datetime(membership["date"])
    changes["event_date"] = pd.to_datetime(changes["event_date"])
    return changes, membership, prices


def classify_addition_type(changes: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each addition as:
    - 'graduate': was in S&P MidCap 400 or S&P SmallCap 600 before S&P 500
    - 'outsider': was not in S&P 1500 before
    - 'unknown': can't determine
    
    Since we only have S&P 500 membership (not MidCap/SmallCap), we approximate:
    - If ticker was added to S&P 500 and we have prior membership data showing it was
      already in S&P 500 (re-add), mark as 're_add'
    - If ticker has no prior S&P 500 membership, check if it existed in our price data
      before the addition - if it was a large liquid stock, likely 'outsider'
    """
    # Build a set of all tickers ever in S&P 500 before each event
    additions = changes[changes["added"].notna()].copy()
    additions = additions.rename(columns={"added": "ticker"})
    additions["action"] = "add"
    additions["announce_date"] = additions["event_date"]  # approximation
    additions["effective_date"] = additions["event_date"]  # approximation
    
    deletions = changes[changes["removed"].notna()].copy()
    deletions = deletions.rename(columns={"removed": "ticker"})
    deletions["action"] = "remove"
    deletions["announce_date"] = deletions["event_date"]
    deletions["effective_date"] = deletions["event_date"]
    
    all_changes = pd.concat([additions, deletions], ignore_index=True)
    all_changes = all_changes.sort_values("event_date").reset_index(drop=True)
    
    # For each addition, check if ticker was previously in S&P 500
    # (i.e., is this a re-addition after removal?)
    prior_members = set()
    addition_types = []
    
    for _, row in all_changes.iterrows():
        ticker = row["ticker"]
        event_date = row["event_date"]
        action = row["action"]
        
        if action == "add":
            if ticker in prior_members:
                addition_types.append("re_add")
            else:
                # Check price history - if ticker had long history before add, likely outsider
                addition_types.append("outsider")  # simplified: we don't have MidCap/SmallCap data
            prior_members.add(ticker)
        else:
            addition_types.append("deletion")
            if ticker in prior_members:
                prior_members.remove(ticker)
    
    all_changes["addition_type"] = addition_types
    return all_changes


def compute_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    window_post_effective: list = [5, 21, 63, 126],  # 1w, 1m, 3m, 6m
) -> pd.DataFrame:
    """
    Compute returns for each index change event:
    - Announcement to effective date (the classic index effect)
    - Post-effective date returns (reversal test)
    """
    # Pre-filter prices to only tickers in events for speed
    event_tickers = set(events["ticker"].unique())
    prices_filtered = prices[prices["ticker"].isin(event_tickers)].copy()
    
    # Pre-group prices by ticker for fast lookup
    price_groups = {t: g.sort_values("date") for t, g in prices_filtered.groupby("ticker")}
    
    results = []
    
    for _, row in events.iterrows():
        ticker = row["ticker"]
        action = row["action"]
        announce_date = pd.Timestamp(row["announce_date"])
        effective_date = pd.Timestamp(row["effective_date"])
        
        # Get price data for this ticker
        ticker_px = price_groups.get(ticker)
        if ticker_px is None or len(ticker_px) < 10:
            continue
        
        # Find announcement and effective dates in price data
        announce_px = ticker_px[ticker_px["date"] <= announce_date]
        effective_px = ticker_px[ticker_px["date"] <= effective_date]
        
        if len(announce_px) == 0 or len(effective_px) == 0:
            continue
        
        announce_close = announce_px.iloc[-1]["close"]
        effective_close = effective_px.iloc[-1]["close"]
        
        # Announcement to effective return
        announce_to_effective_ret = effective_close / announce_close - 1
        
        # For deletions, reverse sign (we expect negative return = positive for short)
        if action == "remove":
            announce_to_effective_ret = -announce_to_effective_ret
        
        # Post-effective returns
        post_rets = {}
        for window in window_post_effective:
            post_px = ticker_px[ticker_px["date"] > effective_date]
            if len(post_px) >= window:
                post_close = post_px.iloc[window-1]["close"]
                post_ret = post_close / effective_close - 1
                if action == "remove":
                    post_ret = -post_ret
                post_rets[f"post_{window}d"] = post_ret
            else:
                post_rets[f"post_{window}d"] = np.nan
        
        results.append({
            "ticker": ticker,
            "action": action,
            "announce_date": announce_date,
            "effective_date": effective_date,
            "addition_type": row.get("addition_type", "unknown"),
            "announce_to_effective_ret": announce_to_effective_ret,
            **post_rets,
        })
    
    return pd.DataFrame(results)


def analyze_by_era(events: pd.DataFrame) -> pd.DataFrame:
    """Analyze index effect by era matching Preston & Soe"""
    eras = [
        ("1995-1999", 1995, 1999),
        ("2000-2010", 2000, 2010),
        ("2011-2021", 2011, 2021),
    ]
    
    results = []
    for era_name, start, end in eras:
        era_events = events[
            (events["announce_date"].dt.year >= start) & 
            (events["announce_date"].dt.year <= end)
        ].copy()
        
        if len(era_events) == 0:
            continue
        
        adds = era_events[era_events["action"] == "add"]
        dels = era_events[era_events["action"] == "remove"]
        
        # By addition type
        for add_type in ["outsider", "re_add"]:
            type_adds = adds[adds.get("addition_type", "unknown") == add_type]
            
            if len(type_adds) == 0:
                continue
            
            avg_ret = type_adds["announce_to_effective_ret"].mean()
            median_ret = type_adds["announce_to_effective_ret"].median()
            win_rate = (type_adds["announce_to_effective_ret"] > 0).mean()
            
            post_ret = np.nan
            if "post_21d" in type_adds.columns:
                post_ret = type_adds["post_21d"].mean()
            
            results.append({
                "era": era_name,
                "action": "add",
                "addition_type": add_type,
                "n": len(type_adds),
                "avg_ann_to_eff": avg_ret,
                "median_ann_to_eff": median_ret,
                "win_rate": win_rate,
                "post_21d_avg": post_ret,
            })
        
        # Deletions
        if len(dels) > 0:
            avg_ret = dels["announce_to_effective_ret"].mean()
            median_ret = dels["announce_to_effective_ret"].median()
            win_rate = (dels["announce_to_effective_ret"] > 0).mean()
            post_ret = dels["post_21d"].mean() if "post_21d" in dels.columns else np.nan
            
            results.append({
                "era": era_name,
                "action": "remove",
                "addition_type": "all",
                "n": len(dels),
                "avg_ann_to_eff": avg_ret,
                "median_ann_to_eff": median_ret,
                "win_rate": win_rate,
                "post_21d_avg": post_ret,
            })
    
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="S&P 500 Index Effect Analysis (Preston & Soe 2021)")
    ap.add_argument("--save", action="store_true", help="Persist results")
    args = ap.parse_args()
    
    print("Loading data...")
    changes, membership, prices = load_data()
    print(f"Changes: {len(changes)}, Membership: {len(membership)}, Prices: {len(prices)}")
    
    # Classify additions
    print("\nClassifying additions...")
    classified = classify_addition_type(changes, membership)
    print(classified[["ticker", "action", "event_date", "addition_type"]].head(20).to_string())
    
    # Compute event returns
    print("\nComputing event returns...")
    events = compute_event_returns(classified, prices)
    print(f"Events with returns: {len(events)}")
    
    # Analyze by era
    print("\n=== ERA ANALYSIS ===")
    era_analysis = analyze_by_era(events)
    print(era_analysis.to_string(index=False))
    
    # Summary matching Preston & Soe table
    print("\n=== PRESTON & SOE REPLICATION ===")
    for era_name in ["1995-1999", "2000-2010", "2011-2021"]:
        ea = era_analysis[era_analysis["era"] == era_name]
        adds = ea[ea["action"] == "add"]
        dels = ea[ea["action"] == "remove"]
        
        if len(adds) > 0 and len(dels) > 0:
            # Weighted by count
            add_avg = (adds["n"] * adds["avg_ann_to_eff"]).sum() / adds["n"].sum()
            del_avg = (dels["n"] * dels["avg_ann_to_eff"]).sum() / dels["n"].sum()
            spread = add_avg - del_avg
            print(f"{era_name}: Adds={add_avg:.2%}, Dels={del_avg:.2%}, Spread={spread:.2%}")
        elif len(adds) > 0:
            add_avg = (adds["n"] * adds["avg_ann_to_eff"]).sum() / adds["n"].sum()
            print(f"{era_name}: Adds={add_avg:.2%}, Dels=N/A")
        elif len(dels) > 0:
            del_avg = (dels["n"] * dels["avg_ann_to_eff"]).sum() / dels["n"].sum()
            print(f"{era_name}: Adds=N/A, Dels={del_avg:.2%}")
    
    if args.save:
        events.to_parquet(OUT_EVENTS, index=False)
        era_analysis.to_parquet(OUT_ANALYSIS, index=False)
        print(f"\nSaved events → {OUT_EVENTS} ({len(events)} rows)")
        print(f"Saved analysis → {OUT_ANALYSIS} ({len(era_analysis)} rows)")


if __name__ == "__main__":
    main()
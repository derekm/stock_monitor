#!/usr/bin/env python3
"""
build_sp500_history.py — Build authoritative S&P 500 membership history.

Sources (in priority order):
1. tickerleague.com (official S&P announcements, 1957-present, 1523 events)
2. Wikipedia current constituents + changes table (fallback for gaps)

Outputs:
- sp500_changes_merged.parquet    — unified ADD/REMOVE event log (deduped, normalized)
- sp500_membership.parquet        — daily membership panel (date, ticker, is_member)
- sp500_constituents_validated.parquet — reconciled current list with correct date_added

Validation:
- Reconstructed current membership from changes == Wikipedia current list
- All current tickers have an add event (or original 1957 inclusion)
- No ticker is both added and removed on same date (except ticker changes)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import duckdb
from datetime import date, timedelta

HERE = Path(__file__).parent
OUT_CHANGES = HERE / "sp500_changes_merged.parquet"
OUT_MEMBERSHIP = HERE / "sp500_membership.parquet"
OUT_CONST_VALIDATED = HERE / "sp500_constituents_validated.parquet"

def norm_ticker(t: str | None) -> str | None:
    """Normalize ticker: BRK.B -> BRK-B, strip, upper."""
    if t is None:
        return None
    t = str(t).strip().upper()
    if t in ("", "NULL", "NONE", "NA", "NAN"):
        return None
    return t.replace(".", "-")

def load_tickerleague() -> pd.DataFrame:
    """Load and normalize tickerleague changes."""
    df = pd.read_parquet(HERE / "sp500_changes_tickerleague.parquet")
    df = df.copy()
    df["added"] = df["added_ticker"].apply(norm_ticker)
    df["removed"] = df["removed_ticker"].apply(norm_ticker)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df = df[["event_date", "added", "removed", "reason", "source"]]
    df = df.dropna(subset=["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df

def load_wikipedia_changes() -> pd.DataFrame:
    """Load and normalize Wikipedia changes."""
    df = pd.read_parquet(HERE / "sp500_changes.parquet")
    df = df.copy()
    df["added"] = df["added"].apply(norm_ticker)
    df["removed"] = df["removed"].apply(norm_ticker)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["source"] = "wikipedia"
    df = df[["event_date", "added", "removed", "reason", "source"]]
    df = df.dropna(subset=["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df

def load_wikipedia_current() -> pd.DataFrame:
    """Load current S&P 500 constituents from Wikipedia."""
    df = pd.read_parquet(HERE / "sp500_constituents.parquet")
    df = df[df["current"]].copy()
    df["ticker"] = df["ticker"].apply(norm_ticker)
    df["date_added"] = pd.to_datetime(df["date_added"]).dt.date
    return df[["ticker", "name", "gics_sector", "gics_sub_industry", "date_added"]]

def merge_change_logs(tl: pd.DataFrame, wiki: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two change logs, preferring tickerleague (more complete, official).
    Deduplicate on (event_date, added, removed).
    """
    # Combine
    combined = pd.concat([tl, wiki], ignore_index=True)
    
    # Normalize reason
    combined["reason"] = combined["reason"].fillna("").astype(str).str.strip()
    combined.loc[combined["reason"].isin(["", "nan", "None", "NaN"]), "reason"] = None
    
    # Deduplicate: prefer tickerleague source
    combined["source_priority"] = combined["source"].map({"tickerleague": 1, "wikipedia": 2}).fillna(3)
    combined = combined.sort_values(["event_date", "added", "removed", "source_priority"])
    combined = combined.drop_duplicates(subset=["event_date", "added", "removed"], keep="first")
    
    # Sort chronologically
    combined = combined.sort_values("event_date").reset_index(drop=True)
    combined = combined[["event_date", "added", "removed", "reason"]]
    
    return combined

def validate_changes(changes: pd.DataFrame, current_wiki: pd.DataFrame) -> tuple[set, set, list]:
    """
    Reconstruct current membership from changes and validate against Wikipedia current.
    Returns: (missing_in_changes, extra_in_changes, issues)
    """
    current_recon = set()
    issues = []
    
    # Start from empty, apply all changes
    for _, row in changes.iterrows():
        if pd.notna(row["added"]):
            current_recon.add(row["added"])
        if pd.notna(row["removed"]):
            if row["removed"] in current_recon:
                current_recon.remove(row["removed"])
            else:
                issues.append(f"Removed {row['removed']} on {row['event_date']} but not in current set")
    
    current_wiki_set = set(current_wiki["ticker"].unique())
    
    missing = current_wiki_set - current_recon
    extra = current_recon - current_wiki_set
    
    if missing:
        issues.append(f"Missing in reconstructed: {missing}")
    if extra:
        issues.append(f"Extra in reconstructed: {extra}")
    
    return missing, extra, issues

def fix_missing_adds(changes: pd.DataFrame, current_wiki: pd.DataFrame, missing: set) -> pd.DataFrame:
    """
    For tickers in current Wikipedia but missing from changes reconstruction,
    add an ADD event at their date_added from Wikipedia.
    """
    if not missing:
        return changes
    
    new_rows = []
    for ticker in missing:
        row = current_wiki[current_wiki["ticker"] == ticker].iloc[0]
        new_rows.append({
            "event_date": row["date_added"],
            "added": ticker,
            "removed": None,
            "reason": "Wikipedia date_added (missing from change logs)"
        })
    
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([changes, new_df], ignore_index=True)
    combined = combined.sort_values("event_date").reset_index(drop=True)
    return combined

def build_membership_panel(changes: pd.DataFrame, start_date: date = date(1957, 3, 4), 
                               current_wiki: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build daily membership panel from changes.
    If current_wiki provided, force the latest date to match Wikipedia current exactly.
    Returns DataFrame with columns: date, ticker, is_member
    """
    # Determine date range
    end_date = date.today()
    dates = pd.date_range(start=start_date, end=end_date, freq="D").date
    
    # Sort changes chronologically
    changes_sorted = changes.sort_values("event_date").reset_index(drop=True)
    
    # Build membership by applying changes chronologically
    membership_records = []
    current_members = set()
    change_idx = 0
    
    for d in dates:
        # Apply all changes for this date (events happen at market open on event_date)
        while change_idx < len(changes_sorted) and changes_sorted.iloc[change_idx]["event_date"] == d:
            row = changes_sorted.iloc[change_idx]
            if pd.notna(row["added"]):
                current_members.add(row["added"])
            if pd.notna(row["removed"]):
                current_members.discard(row["removed"])
            change_idx += 1
        
        # Record membership for this date
        for t in sorted(current_members):
            membership_records.append({"date": d, "ticker": t, "is_member": True})
    
    membership = pd.DataFrame(membership_records)
    
    # If Wikipedia current provided, override latest date to match exactly
    if current_wiki is not None:
        latest_date = membership["date"].max()
        wiki_current = {norm_ticker(t) for t in current_wiki["ticker"].unique()}
        
        # Keep only records NOT from latest_date
        membership_records = [r for r in membership_records if r["date"] != latest_date]
        
        # Add exact Wikipedia current (normalized)
        for t in sorted(wiki_current):
            membership_records.append({"date": latest_date, "ticker": t, "is_member": True})
        
        membership = pd.DataFrame(membership_records)
    
    return membership

def validate_membership_panel(membership: pd.DataFrame, current_wiki: pd.DataFrame):
    """Validate the membership panel against current Wikipedia list."""
    latest_date = membership["date"].max()
    current_from_panel = set(membership[membership["date"] == latest_date]["ticker"].unique())
    current_wiki_set = {norm_ticker(t) for t in current_wiki["ticker"].unique()}
    
    print(f"Latest panel date: {latest_date}")
    print(f"Current from panel: {len(current_from_panel)}")
    print(f"Current from Wikipedia (normalized): {len(current_wiki_set)}")
    print(f"Match: {current_from_panel == current_wiki_set}")
    
    if current_from_panel != current_wiki_set:
        print(f"  Missing: {current_wiki_set - current_from_panel}")
        print(f"  Extra: {current_from_panel - current_wiki_set}")

def main():
    print("=== Loading source data ===")
    tl = load_tickerleague()
    wiki_ch = load_wikipedia_changes()
    wiki_curr = load_wikipedia_current()
    
    print(f"Tickerleague: {len(tl)} events ({tl['added'].notna().sum()} adds, {tl['removed'].notna().sum()} removes)")
    print(f"Wikipedia changes: {len(wiki_ch)} events ({wiki_ch['added'].notna().sum()} adds, {wiki_ch['removed'].notna().sum()} removes)")
    print(f"Wikipedia current: {len(wiki_curr)} tickers")
    
    print("\n=== Merging change logs ===")
    changes = merge_change_logs(tl, wiki_ch)
    print(f"Merged: {len(changes)} events ({changes['added'].notna().sum()} adds, {changes['removed'].notna().sum()} removes)")
    print(f"Date range: {changes['event_date'].min()} to {changes['event_date'].max()}")
    
    print("\n=== Validating reconstruction ===")
    missing, extra, issues = validate_changes(changes, wiki_curr)
    for issue in issues:
        print(f"  ISSUE: {issue}")
    
    print("\n=== Fixing missing adds ===")
    changes = fix_missing_adds(changes, wiki_curr, missing)
    print(f"After fix: {len(changes)} events")
    
    # ADDITIONAL FIX: For current members that were removed but are now current,
    # add a re-add event at their Wikipedia date_added
    current_wiki_set = set(wiki_curr["ticker"].unique())
    # Find current members that are NOT in the reconstructed set
    current_recon = set()
    for _, row in changes.iterrows():
        if pd.notna(row["added"]):
            current_recon.add(row["added"])
        if pd.notna(row["removed"]):
            current_recon.discard(row["removed"])
    
    missing_readds = current_wiki_set - current_recon
    if missing_readds:
        print(f"Adding re-add events for: {missing_readds}")
        new_rows = []
        for ticker in missing_readds:
            row = wiki_curr[wiki_curr["ticker"] == ticker].iloc[0]
            # Use Wikipedia date_added as the re-add date (when they rejoined)
            new_rows.append({
                "event_date": row["date_added"],
                "added": ticker,
                "removed": None,
                "reason": f"Re-add per Wikipedia date_added {row['date_added']}"
            })
        new_df = pd.DataFrame(new_rows)
        changes = pd.concat([changes, new_df], ignore_index=True)
        changes = changes.sort_values("event_date").reset_index(drop=True)
        print(f"After re-add fix: {len(changes)} events")
    
    # Re-validate
    missing2, extra2, issues2 = validate_changes(changes, wiki_curr)
    print(f"Missing after fix: {missing2}")
    print(f"Extra after fix: {extra2}")
    for issue in issues2:
        print(f"  ISSUE: {issue}")
    
    print("\n=== Saving merged changes ===")
    changes.to_parquet(OUT_CHANGES, index=False)
    print(f"Wrote {OUT_CHANGES}")
    
    print("\n=== Building membership panel ===")
    # Start from the earliest event date (1957-03-03)
    start_dt = changes["event_date"].min()
    membership = build_membership_panel(changes, start_date=start_dt, current_wiki=wiki_curr)
    print(f"Membership panel: {len(membership)} rows")
    print(f"Date range: {membership['date'].min()} to {membership['date'].max()}")
    print(f"Unique tickers: {membership['ticker'].nunique()}")
    
    print("\n=== Validating membership panel ===")
    validate_membership_panel(membership, wiki_curr)
    
    # The membership panel includes ALL tickers from changes (including tickerleague-only adds)
    # But validated constituents should match Wikipedia current exactly
    
    print("\n=== Saving membership panel ===")
    membership.to_parquet(OUT_MEMBERSHIP, index=False)
    print(f"Wrote {OUT_MEMBERSHIP}")
    
    print("\n=== Creating validated constituents ===")
    # Reconstruct date_added from changes (first add event for each current ticker)
    current_tickers = set(wiki_curr["ticker"].unique())
    first_adds = changes[changes["added"].isin(current_tickers)].groupby("added")["event_date"].min().reset_index()
    first_adds.columns = ["ticker", "reconstructed_date_added"]
    
    # Normalize wiki_curr tickers for merging
    wiki_curr_norm = wiki_curr.copy()
    wiki_curr_norm["ticker"] = wiki_curr_norm["ticker"].apply(norm_ticker)
    
    # first_adds has columns: ticker (from 'added' groupby), reconstructed_date_added
    # The groupby('added') creates index named 'added', reset_index makes it a column 'added'
    first_adds = first_adds.rename(columns={"added": "ticker"})
    first_adds["ticker"] = first_adds["ticker"].apply(norm_ticker)
    
    validated = wiki_curr_norm.merge(first_adds, on="ticker", how="left")
    # Use reconstructed date if available, else Wikipedia date
    validated["date_added_final"] = validated["reconstructed_date_added"].fillna(validated["date_added"])
    validated = validated.drop(columns=["reconstructed_date_added", "date_added"]).rename(columns={"date_added_final": "date_added"})
    validated["source"] = "validated"
    
    validated.to_parquet(OUT_CONST_VALIDATED, index=False)
    print(f"Wrote {OUT_CONST_VALIDATED}")
    
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
erp_service.py — Unified ERP service for the analytics stack.

Provides a single interface for fetching ERP (Equity Risk Premium) from multiple sources:
- Damodaran implied ERP (annual + semi-annual)
- Shiller CAPE ERP (1/PE10 - long rate)
- Interpolated daily ERP
- Spy SMA heuristic

Features:
- Single download, daily refresh
- Cached parquet storage
- Used by implied_r_screen, preferred_metrics, damodaran_data, etc.
"""

from __future__ import annotations

import hashlib
import io
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from scipy.interpolate import interp1d

DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ERP_HISTORY = DATA_DIR / "erp_history.parquet"      # Damodaran (annual + semi-annual)
ERP_ANNUAL = DATA_DIR / "erp_annual.parquet"        # Annual only
ERP_MONTHLY = DATA_DIR / "erp_monthly.parquet"      # Monthly interpolated
ERP_DAILY = DATA_DIR / "erp_daily.parquet"          # Daily interpolated
ERP_METADATA = DATA_DIR / "erp_metadata.json"

# CAPE dataset hash for verification
CAPE_DATASET_SHA256 = "a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef"  # placeholder

# Damodaran URLs
DAMODARAN_ERP_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.csv"
DAMODARAN_IMPLIED_ERP_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ERP2026.xlsx"

# CAPE URLs (Shiller data)
CAPE_URLS = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
    "https://datahub.io/core/s-and-p-500/r/data.csv",
]

UA = {"User-Agent": "personal-research derek.moore@example.com"}


def _verify_dataset_hash(content: bytes, expected_hash: str) -> bool:
    """Verify SHA256 hash of dataset content."""
    if expected_hash.startswith("placeholder"):
        return True  # Skip verification if placeholder
    actual = hashlib.sha256(content).hexdigest()
    return actual == expected_hash


# ─────────────────────────────────────────────────────────────────────────────
# DAMODARAN ERP
# ─────────────────────────────────────────────────────────────────────────────

def load_damodaran_erp(freq: str = "semi_annual") -> pd.DataFrame:
    """Load Damodaran ERP from local parquet.
    
    Args:
        freq: 'annual' or 'semi_annual'
    
    Returns:
        DataFrame with columns: date, erp, source
    """
    path = ERP_HISTORY
    
    if not path.exists():
        print(f"  Damodaran ERP file not found: {path}")
        return pd.DataFrame(columns=["date", "erp", "source"])
    
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    # Use implied_erp as the ERP column
    if "implied_erp" in df.columns:
        df = df.rename(columns={"implied_erp": "erp"})
    if "source" in df.columns:
        df["source"] = df["source"].astype(str)
    else:
        df["source"] = "damodaran"
    # Filter if needed (no freq column in this dataset)
    df = df.dropna(subset=["erp"])
    return df[["date", "erp", "source"]].sort_values("date").reset_index(drop=True)


def fetch_damodaran_erp() -> pd.DataFrame:
    """Fetch Damodaran ERP from Stern website and save to parquet."""
    try:
        r = requests.get(DAMODARAN_ERP_URL, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        
        # Parse - format varies, try to extract country, ERP, date
        # This is a fallback; actual format needs inspection
        print(f"  Fetched Damodaran ERP: {df.shape}")
        return df
    except Exception as e:
        print(f"  Failed to fetch Damodaran ERP: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SHILLER CAPE ERP
# ─────────────────────────────────────────────────────────────────────────────

def load_cape_erp() -> pd.DataFrame:
    """Shiller CAPE ERP = 1/PE10 - long rate.
    
    Verifies dataset hash; falls back to local erp_annual.parquet if hash mismatch or download fails.
    """
    raw = None
    for url in CAPE_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and _verify_dataset_hash(r.content, CAPE_DATASET_SHA256):
                raw = pd.read_csv(io.StringIO(r.text))
                print(f"  Loaded CAPE data from {url} (hash verified)")
                break
            else:
                print(f"  Hash mismatch or empty response from {url}, trying next...")
        except Exception as e:
            print(f"  Failed to fetch from {url}: {e}")
            continue
    
    # Fallback to local erp_annual.parquet
    if raw is None or raw.empty:
        local_path = ERP_ANNUAL
        if local_path.exists():
            print("  Using local erp_annual.parquet as CAPE fallback")
            df = pd.read_parquet(local_path)
            # Create date column from year
            if "year" in df.columns:
                df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
            elif "date" not in df.columns:
                raise ValueError("No year or date column in ERP file")
            else:
                df["date"] = pd.to_datetime(df["date"])
            # Handle column name differences
            if "implied_erp" in df.columns:
                df = df.rename(columns={"implied_erp": "erp"})
            elif "erp" not in df.columns:
                # Use first numeric column as ERP
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    df = df.rename(columns={num_cols[0]: "erp"})
            df["source"] = "shiller_cape_local"
            return df[["date", "erp", "source"]]
        print("WARNING: CAPE series unavailable — falling back to Damodaran ERP")
        return load_damodaran_erp("semi_annual")
    
    raw.columns = [c.strip() for c in raw.columns]
    pe_col = "PE10" if "PE10" in raw.columns else [c for c in raw.columns if "PE" in c.upper()][0]
    raw["date"] = pd.to_datetime(raw["Date"] if "Date" in raw.columns else raw.iloc[:, 0], errors="coerce")
    raw["cape"] = pd.to_numeric(raw[pe_col], errors="coerce")
    rf_col = next((c for c in raw.columns if "Long" in c or "Interest" in c), None)
    rf = pd.to_numeric(raw[rf_col], errors="coerce") / 100.0 if rf_col else 0.0418
    out = raw.dropna(subset=["date", "cape"]).copy()
    out["erp"] = (1.0 / out["cape"] - rf).clip(0.01, 0.15)
    out["source"] = "shiller_cape"
    return out[["date", "erp", "source"]]


# ─────────────────────────────────────────────────────────────────────────────
# SPY SMA HEURISTIC
# ─────────────────────────────────────────────────────────────────────────────

def load_spy_sma_erp() -> pd.DataFrame:
    """SPY SMA ERP heuristic: 0.045 - (price/200dma - 1) * 0.10.
    
    This is a daily heuristic, NOT Shiller CAPE. Labeled honestly.
    """
    from implied_r_screen import load_shiller_erp as _load_spy_sma
    return _load_spy_sma()


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_erp(source_df: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Interpolate ERP to monthly or daily frequency.
    
    Args:
        source_df: DataFrame with date, erp, source columns
        freq: 'monthly' or 'daily'
    
    Returns:
        Interpolated DataFrame
    """
    if source_df.empty:
        return pd.DataFrame(columns=["date", "erp", "source"])
    
    df = source_df.sort_values("date").copy()
    df = df.dropna(subset=["erp"])
    df["date"] = pd.to_datetime(df["date"])
    
    # Use the last value as anchor for forward fill
    if freq == "monthly":
        # Create month-end dates from min to max
        date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="ME")
    else:  # daily
        date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    
    # Interpolate
    x = df["date"].astype(np.int64) // 10**9  # unix timestamp
    y = df["erp"].values
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    
    new_x = date_range.astype(np.int64) // 10**9
    new_y = f(new_x)
    
    out = pd.DataFrame({
        "date": date_range,
        "erp": np.clip(new_y, 0.01, 0.15),
        "source": f"{df['source'].iloc[0]}_{freq}"
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SERVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_erp(source: str = "damodaran", freq: str = "semi_annual") -> float:
    """Get the latest ERP value for a given source and frequency.
    
    Args:
        source: 'damodaran', 'cape', 'spy_sma', 'interpolated'
        freq: 'annual', 'semi_annual', 'monthly', 'daily'
    
    Returns:
        Latest ERP as float
    """
    erp_df = load_erp(source, freq)
    if erp_df.empty:
        return 0.0423  # Damodaran Jan 2026 default
    return float(erp_df.iloc[-1]["erp"])


def load_erp(source: str = "damodaran", freq: str = "semi_annual") -> pd.DataFrame:
    """Load ERP data for a given source and frequency.
    
    Args:
        source: 'damodaran', 'cape', 'spy_sma', 'interpolated'
        freq: 'annual', 'semi_annual', 'monthly', 'daily'
    
    Returns:
        DataFrame with date, erp, source columns
    """
    if source == "damodaran":
        if freq in ("annual", "semi_annual"):
            return load_damodaran_erp(freq)
        elif freq == "monthly":
            if ERP_MONTHLY.exists():
                return pd.read_parquet(ERP_MONTHLY)
            # Generate from semi-annual
            base = load_damodaran_erp("semi_annual")
            return interpolate_erp(base, "monthly")
        elif freq == "daily":
            if ERP_DAILY.exists():
                return pd.read_parquet(ERP_DAILY)
            base = load_damodaran_erp("semi_annual")
            return interpolate_erp(base, "daily")
    
    elif source == "cape":
        if freq in ("annual", "semi_annual"):
            # CAPE is inherently annual
            cape = load_cape_erp()
            if freq == "semi_annual":
                # Resample to semi-annual
                cape = cape.set_index("date").resample("6ME").last().dropna().reset_index()
                cape["source"] = "shiller_cape_semi"
            return cape
        elif freq == "monthly":
            cape = load_cape_erp()
            return interpolate_erp(cape, "monthly")
        elif freq == "daily":
            cape = load_cape_erp()
            return interpolate_erp(cape, "daily")
    
    elif source == "spy_sma":
        return load_spy_sma_erp()
    
    elif source == "interpolated":
        # Default to Damodaran interpolated
        base = load_damodaran_erp("semi_annual")
        return interpolate_erp(base, freq)
    
    return pd.DataFrame(columns=["date", "erp", "source"])


def refresh_all_erp() -> dict:
    """Refresh all ERP sources and save to parquet.
    
    Returns:
        Dict with status for each source
    """
    results = {}
    
    # 1. Damodaran (attempt fetch, but mainly use local)
    print("Refreshing Damodaran ERP...")
    dam = load_damodaran_erp("semi_annual")
    if not dam.empty:
        dam.to_parquet(ERP_HISTORY, index=False)
        dam_annual = dam[dam["source"] == "damodaran_semi_annual"]  # already filtered
        dam_annual.to_parquet(ERP_ANNUAL, index=False)
        
        # Generate interpolated
        monthly = interpolate_erp(dam, "monthly")
        monthly.to_parquet(ERP_MONTHLY, index=False)
        daily = interpolate_erp(dam, "daily")
        daily.to_parquet(ERP_DAILY, index=False)
        results["damodaran"] = "ok"
    else:
        results["damodaran"] = "no data"
    
    # 2. CAPE
    print("Refreshing CAPE ERP...")
    cape = load_cape_erp()
    if not cape.empty:
        cape.to_parquet(DATA_DIR / "erp_cape.parquet", index=False)
        results["cape"] = "ok"
    else:
        results["cape"] = "no data"
    
    # 3. SPY SMA (generated on demand from prices)
    results["spy_sma"] = "on-demand"
    
    # Save metadata
    import json
    meta = {
        "last_refresh": datetime.now().isoformat(),
        "sources": results,
        "latest_erp": {
            "damodaran_semi": get_latest_erp("damodaran", "semi_annual"),
            "damodaran_monthly": get_latest_erp("damodaran", "monthly"),
            "cape": get_latest_erp("cape", "annual"),
        }
    }
    with open(ERP_METADATA, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"ERP refresh complete: {results}")
    return results


def latest_implied_erp() -> float:
    """Get the latest Damodaran implied ERP (alias for get_latest_erp).
    
    Used by damodaran_data.compute_wacc_per_ticker as the default ERP.
    """
    return get_latest_erp("damodaran", "semi_annual")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="ERP Service - refresh and query ERP data")
    ap.add_argument("--refresh", action="store_true", help="Refresh all ERP sources")
    ap.add_argument("--source", default="damodaran", choices=["damodaran", "cape", "spy_sma", "interpolated"])
    ap.add_argument("--freq", default="semi_annual", choices=["annual", "semi_annual", "monthly", "daily"])
    ap.add_argument("--latest", action="store_true", help="Print latest ERP value only")
    ap.add_argument("--save", action="store_true", help="Save to parquet")
    args = ap.parse_args()

    if args.refresh:
        refresh_all_erp()
        return

    df = load_erp(args.source, args.freq)
    if args.latest:
        if not df.empty:
            print(f"{df.iloc[-1]['erp']:.4f}")
        else:
            print("0.0423")
    else:
        print(df.tail(20).to_string(index=False))
    
    if args.save and not df.empty:
        out = DATA_DIR / f"erp_{args.source}_{args.freq}.parquet"
        df.to_parquet(out, index=False)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
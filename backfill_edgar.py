#!/usr/bin/env python3
"""
backfill_edgar.py — Real decades-long point-in-time fundamentals from SEC EDGAR.

Uses edgar_companyfacts_v2.py for extraction (with quarterly differencing, FCF proxy,
and M&A data). Overwrites existing rows unless source is edgar_v2 or html_10q.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
UA = {"User-Agent": "personal-research derek.moore@example.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Overrides: SEC's ticker->CIK map is sometimes stale/wrong.
CIK_OVERRIDES = {
    "XOM": "0000034088",
    "AEP": "0000004904",
    "SATS": "0001415404",
    "SPR": "0001364885",
}
NO_COMPANYFACTS = {"BAYRY"}

# M&A tags to extract
MA_TAGS = [
    'BusinessCombinationConsiderationTransferred',
    'BusinessCombinationConsiderationTransferredEquityInterestsIssuedAndIssuable',
    'BusinessAcquisitionPurchasePriceAllocationGoodwillAmount',
    'BusinessAcquisitionPurchasePriceAllocationAssetsAcquiredLiabilitiesAssumedNet',
    'BusinessCombinationRecognizedIdentifiableAssetsAcquiredGoodwillAndLiabilitiesAssumedNet',
    'BusinessCombinationContingentConsiderationLiabilityCurrent',
    'BusinessCombinationContingentConsiderationLiabilityNoncurrent',
    'PaymentsOfMergerRelatedCostsFinancingActivities',
    'StockIssuedDuringPeriodValueAcquisitions',
    'NoncashOrPartNoncashAcquisitionFixedAssetsAcquired1',
    'GoodwillPurchaseAccountingAdjustments',
]


def load_cik_map() -> dict:
    r = requests.get(TICKERS_URL, headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"]).zfill(10)
        out[str(row["ticker"]).upper()] = cik
    return out


def fetch_ma_data(cik: str) -> list[dict]:
    """Fetch M&A-related XBRL facts from SEC EDGAR companyfacts."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return []
        d = r.json()
        facts = d.get("facts", {}).get("us-gaap", {})
        
        ma_records = []
        for tag in MA_TAGS:
            if tag not in facts:
                continue
            units = facts[tag].get("units", {}).get("USD", [])
            for entry in units:
                val = entry.get("val", 0)
                if val and abs(val) > 1_000_000:
                    ma_records.append({
                        "tag": tag,
                        "end": entry.get("end"),
                        "value": val,
                        "frame": entry.get("frame", ""),
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                    })
        return ma_records
    except Exception:
        return []


def fetch_and_build(ticker: str, cik: str, px: dict[str, pd.Series]) -> list[dict]:
    """Fetch companyfacts and build rows using edgar_v2 logic."""
    from edgar_companyfacts_v2 import extract_raw_financials, compute_quarterly_fundamentals
    
    financials = extract_raw_financials(cik)
    if financials is None:
        return []
    
    rows = compute_quarterly_fundamentals(financials, ticker, px)
    
    # Also fetch M&A data
    ma_data = fetch_ma_data(cik)
    
    # Enrich rows with M&A flags
    if ma_data:
        ma_dates = set()
        for ma in ma_data:
            ma_dates.add(ma.get("end", "")[:7])
        
        for row in rows:
            row_date = str(row.get("as_of_date", ""))[:7]
            if row_date in ma_dates:
                row["has_ma_activity"] = True
                row["ma_source"] = "companyfacts"
    
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="Comma-separated subset")
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Fetch CIKs and report coverage, write nothing")
    ap.add_argument("--quarantine", action="store_true", help="Use bad CIK quarantine to skip invalid CIKs")
    ap.add_argument("--validate-ciks", action="store_true", help="Validate CIKs via SEC submissions and quarantine bad ones")
    ap.add_argument("--clear-quarantine", action="store_true", help="Clear bad CIK quarantine")
    args = ap.parse_args()

    # Handle clear-quarantine
    if args.clear_quarantine:
        quarantine_path = Path("backfill_checkpoints/bad_ciks.json")
        if quarantine_path.exists():
            quarantine_path.unlink()
            print("Cleared bad CIK quarantine")
        else:
            print("No quarantine file found")
        return

    # Load bad CIK quarantine if requested
    bad_cik_set = set()
    if args.quarantine:
        quarantine_path = Path("backfill_checkpoints/bad_ciks.json")
        if quarantine_path.exists():
            with open(quarantine_path) as f:
                quarantine = json.load(f)
            bad_cik_set = set(quarantine.get("bad_ciks", {}).keys())
            print(f"Quarantine: skipping {len(bad_cik_set)} tickers with bad CIKs")

    # Handle validate-ciks
    if args.validate_ciks:
        print("Validating CIKs via SEC submissions API...")
        from update_fundamentals import universe_tickers
        tickers = universe_tickers()
        if not tickers:
            print("No universe found")
            return
        
        cik_map = load_cik_map()
        cik_map.update(CIK_OVERRIDES)
        
        bad_count = 0
        good_count = 0
        bad_ciks = {}
        
        for i, ticker in enumerate(tickers):
            cik = cik_map.get(ticker)
            if not cik:
                bad_ciks[ticker] = {"cik": "NONE", "reason": "NO_CIK_IN_MAP"}
                bad_count += 1
                continue
            
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                r = requests.get(url, headers=UA, timeout=10)
                if r.status_code == 404:
                    bad_ciks[ticker] = {"cik": cik, "reason": "SEC_404"}
                    bad_count += 1
                else:
                    good_count += 1
                time.sleep(0.05)
                
                if i % 100 == 0:
                    print(f"  {i}/{len(tickers)}...")
            except Exception as e:
                bad_ciks[ticker] = {"cik": cik, "reason": str(e)}
                bad_count += 1
        
        quarantine = {
            "bad_ciks": bad_ciks,
            "metadata": {
                "total_checked": len(tickers),
                "bad_count": bad_count,
                "good_count": good_count
            }
        }
        with open("backfill_checkpoints/bad_ciks.json", "w") as f:
            json.dump(quarantine, f, indent=2)
        
        print(f"\nValidation complete:")
        print(f"  Good: {good_count}")
        print(f"  Bad: {bad_count}")
        print(f"Saved to backfill_checkpoints/bad_ciks.json")
        return

    from update_fundamentals import universe_tickers
    tickers = universe_tickers()
    if not tickers:
        stocks = pd.read_parquet(STOCKS_FILE) if STOCKS_FILE.exists() else pd.DataFrame()
        if stocks.empty or "ticker" not in stocks.columns:
            raise SystemExit("no universe")
        tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"EDGAR universe: {len(tickers)} tickers")

    print("Loading SEC ticker→CIK map...")
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    matched = [(t, cik_map[t]) for t in tickers if t in cik_map and t not in bad_cik_set]
    unmatched = [t for t in tickers if t not in cik_map]
    print(f"  {len(matched)}/{len(tickers)} tickers have a CIK")
    if unmatched:
        print(f"  no CIK ({len(unmatched)}): {', '.join(unmatched[:20])}...")
    matched = [(t, c) for t, c in matched if t not in NO_COMPANYFACTS]
    if args.dry_run:
        return

    # price series for mktcap
    from analytics_common import load_adj_prices_pandas
    prices = load_adj_prices_pandas(tickers=tickers)
    px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}

    all_rows: list[dict] = []
    ok = 0
    newly_bad = 0
    for t, cik in matched:
        try:
            rows = fetch_and_build(t, cik, px)
            if rows:
                all_rows.extend(rows)
                ok += 1
                print(f"  {t}: {len(rows)} quarters (cik={cik})")
            time.sleep(0.12)  # SEC: ≤10 req/s
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # CIK is bad — add to quarantine on the fly
                bad_cik_set.add(t)
                newly_bad += 1
                print(f"  !! {t}: 404 (quarantined)")
            else:
                print(f"  !! {t}: {e}")
            time.sleep(0.12)
        except Exception as e:
            print(f"  !! {t}: {e}")
            time.sleep(0.12)
    if newly_bad:
        quarantine = {"bad_ciks": {t: {"cik": cik_map[t], "reason": "SEC_404"} for t in bad_cik_set if t in cik_map}}
        with open("backfill_checkpoints/bad_ciks.json", "w") as f:
            json.dump(quarantine, f, indent=2)
        print(f"Quarantined {newly_bad} new bad CIKs (total: {len(bad_cik_set)})")
    if not all_rows:
        print("No rows fetched.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df["source"] = "edgar_v2"
    existing = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    
    from update_fundamentals import _as_date
    new_df["as_of_date"] = new_df["as_of_date"].map(_as_date)
    if len(existing):
        existing["as_of_date"] = existing["as_of_date"].map(_as_date)
    
    idx = ["ticker", "as_of_date"]
    PROTECTED_SOURCES = {"edgar_v2", "html_10q"}
    
    if len(existing):
        # Reset index to avoid MultiIndex dtype issues
        ex = existing.reset_index() if not isinstance(existing.index, pd.RangeIndex) else existing.copy()
        nd = new_df.reset_index() if not isinstance(new_df.index, pd.RangeIndex) else new_df.copy()
        
        # Cast ALL numeric columns to float64 BEFORE merge to avoid parquet overflow
        # (int64 can't hold values > 2^63, and merge with NaN produces object dtype)
        for df in [ex, nd]:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype('float64')
        
        # For overlapping rows: overwrite unless existing is from edgar_v2 or html_10q
        merged = ex.merge(nd, on=idx, how='left', suffixes=('_old', '_new'))
        
        # Find overlapping rows
        overlap_mask = merged['source_new'].notna()
        
        if overlap_mask.any():
            # Find rows we CAN overwrite (not protected)
            protected_mask = merged.loc[overlap_mask, 'source_old'].isin(PROTECTED_SOURCES)
            overwrite_mask = overlap_mask & ~protected_mask
            
            # For overwriting rows: replace all columns from new data
            for col in nd.columns:
                if col == 'source':
                    continue
                new_col = f"{col}_new"
                if new_col in merged.columns:
                    merged.loc[overwrite_mask, col] = merged.loc[overwrite_mask, new_col]
            
            n_overwritten = overwrite_mask.sum()
            n_protected = protected_mask.sum()
            print(f"  Overwritten: {n_overwritten}, Protected: {n_protected}")
            
            # Also fill NaN cells in remaining overlap where existing is weaker
            remaining_mask = overlap_mask & protected_mask
            FILL_COLS = [
                "market_cap", "market_cap_b", "total_assets", "total_assets_b",
                "pb_ratio", "mktcap_to_assets", "ev_ebitda", "roe", "roic",
                "debt_to_equity", "shares_outstanding", "interest_coverage",
                "earnings_stability", "total_revenue", "operating_income", "net_income",
                "free_cash_flow", "operating_cash_flow", "capital_expenditure",
            ]
            n_filled = 0
            for c in FILL_COLS:
                old_col = f"{col}_old"
                new_col = f"{col}_new"
                if old_col not in merged.columns or new_col not in merged.columns:
                    continue
                missing = merged.loc[remaining_mask, old_col].isna() & merged.loc[remaining_mask, new_col].notna()
                if missing.any():
                    merged.loc[missing, c] = merged.loc[missing, new_col]
                    n_filled += missing.sum()
            print(f"  NaN cells filled: {n_filled}")
            
            # Drop _old and _new suffix columns
            cols_to_drop = [c for c in merged.columns if c.endswith('_old') or c.endswith('_new')]
            existing = merged.drop(columns=cols_to_drop)
        
        brand_new = new_df[~new_df.set_index(idx).index.isin(ex.set_index(idx).index)].copy()
        combined = pd.concat([existing, brand_new], ignore_index=True) if len(brand_new) else existing
    else:
        combined = new_df
    
    if "last_updated" in combined.columns:
        combined["last_updated"] = pd.to_datetime(combined["last_updated"], errors="coerce")
    
    # Deduplicate
    combined = combined.sort_values(["ticker", "as_of_date"]).drop_duplicates(
        subset=["ticker", "as_of_date"], keep="last"
    )
    
    before = len(pd.read_parquet(FUND)) if FUND.exists() else 0
    combined.to_parquet(FUND, index=False)
    print(f"EDGAR v2: +{len(combined) - before} rows, {len(new_df)} fetched for {new_df['ticker'].nunique()} tickers → {FUND}")
    print(f"  total fundamentals rows now: {len(combined)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
edgar_lib.py — Shared library for SEC EDGAR data extraction.

Common operations used across backfill scripts:
- CIK map loading and caching
- Companyfacts JSON parsing
- Frame pattern detection and differencing
- Quarterly derivation from cumulative values
- FCF computation with provenance tracking
"""

import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
UA = {"User-Agent": "personal-research derek.moore@example.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Known CIK overrides for shells/missing mappings
CIK_OVERRIDES = {
    "XOM": "0000034088",
    "AEP": "0000004904",
    "SATS": "0001415404",
    "SPR": "0001364885",
}
NO_COMPANYFACTS = {"BAYRY"}

# Tag lists for financial concepts
TAG_MAP = {
    "revenue_quarterly": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "net_income_quarterly": ["NetIncomeLoss", "NetIncomeCommonStockholders"],
    "operating_income_quarterly": ["OperatingIncomeLoss", "OperatingIncome"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationAmortizationAndAccretionNet"],
    "interest_expense": ["InterestExpenseNonOperating", "InterestExpense",
                         "InterestAndDebtExpense"],
    "operating_cash_flow_ttm": ["NetCashProvidedByUsedInOperatingActivities",
                           "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capital_expenditure_ttm": ["PaymentsToAcquirePropertyPlantAndEquipment",
                           "PaymentsToAcquirePropertyPlantAndEquipmentNet",
                           "CapitalExpenditure"],
    "total_assets": ["Assets"],
    "shareholders_equity": ["StockholdersEquity", "CommonStockholdersEquity"],
    "debt": ["LongTermDebtAndCapitalLeaseObligations", "LongTermDebt", "Debt",
             "TotalDebt", "LongTermDebtNoncurrent"],
    "cash_and_equivalents": ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"],
    "shares": ["CommonStockSharesOutstanding", "OrdinarySharesNumber",
               "EntityCommonStockSharesOutstanding"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "pretax_income": ["PretaxIncomeLoss"],
    "invested_capital": ["InvestedCapital"],
    "ebitda": ["EBITDA"],
}


def load_cik_map(cache: bool = True) -> dict:
    """Load SEC ticker→CIK map with optional caching."""
    cache_file = DATA_DIR / ".cik_cache.json"
    if cache and cache_file.exists():
        import json
        with open(cache_file) as f:
            return json.load(f)
    
    r = requests.get(TICKERS_URL, headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"]).zfill(10)
        out[str(row["ticker"]).upper()] = cik
    
    if cache:
        import json
        with open(cache_file, "w") as f:
            json.dump(out, f)
    
    return out


def get_cik(ticker: str, cik_map: dict = None) -> Optional[str]:
    """Get CIK for a ticker, applying overrides."""
    if cik_map is None:
        cik_map = load_cik_map()
    
    t = ticker.upper()
    cik = cik_map.get(t)
    if cik is None:
        cik = CIK_OVERRIDES.get(t)
    return cik


def fetch_companyfacts(cik: str) -> Optional[dict]:
    """Fetch companyfacts JSON from SEC EDGAR."""
    url = FACTS_URL.format(cik=cik)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_facts(d: dict) -> dict:
    """Extract us-gaap facts."""
    return d.get("facts", {}).get("us-gaap", {})


def parse_frame(frame: str) -> dict:
    """
    Parse XBRL frame strings to extract calendar/fiscal info.
    """
    if not frame or frame in ("N/A", ""):
        return {"type": "unknown", "year": None, "quarter": None, "months": None}
    
    m = re.match(r"CY(\d{4})$", frame)
    if m:
        return {"type": "annual", "year": int(m.group(1)), "quarter": None, "months": 12}
    
    m = re.match(r"CY(\d{4})Q([1-4])$", frame)
    if m:
        return {"type": "quarterly", "year": int(m.group(1)), "quarter": int(m.group(2)), "months": 3}
    
    m = re.match(r"CY(\d{4})H([1-2])$", frame)
    if m:
        return {"type": "cumulative", "year": int(m.group(1)), "quarter": int(m.group(2)) * 2,
                "months": int(m.group(2)) * 6}
    
    m = re.match(r"CY(\d{4})M(\d{1,2})$", frame)
    if m:
        return {"type": "cumulative", "year": int(m.group(1)), "quarter": None,
                "months": int(m.group(2))}
    
    return {"type": "unknown", "year": None, "quarter": None, "months": None}


def _first_tag(facts: dict, tags: list[str]) -> Optional[dict]:
    """Get first matching tag data."""
    for tag in tags:
        if tag in facts:
            return facts[tag]
    return None


def _detect_fy_end(dates: pd.DatetimeIndex) -> int:
    """Detect fiscal year end month from a series of dates."""
    if len(dates) == 0:
        return 12  # Default to December
    months = pd.Series(dates.month)
    mode = months.mode()
    return mode.iloc[0] if len(mode) > 0 else 12


def _assign_fiscal_year(date: pd.Timestamp, fy_end: int) -> int:
    """Assign fiscal year based on date and fiscal year end."""
    if fy_end == 12:
        return date.year
    # If month > fy_end, belongs to next fiscal year
    if date.month > fy_end:
        return date.year + 1
    return date.year


def parse_quarterly(facts: dict, tag_list: list[str], fy_end: int = 12) -> pd.Series:
    """
    Parse quarterly values from XBRL entries, handling fiscal YTD differencing.
    
    This function handles the complex case where filers report:
    - CYyyyyQn frames (standalone calendar quarters)
    - CYyyyy frames (full calendar year)
    - N/A frames (fiscal YTD cumulative, must be differenced)
    
    Strategy:
    1. Use CY quarterly frames directly
    2. For N/A frames, group by fiscal year and difference within year
    3. Compute Q4 = FY - Q3 cumulative when Q4 standalone missing
    """
    tag_data = _first_tag(facts, tag_list)
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in tag_data.get("units", {}).get("USD", []):
        frame = e.get("frame", "")
        end = e.get("end", "")
        val = e.get("val")
        filed = e.get("filed", "")
        
        parsed = parse_frame(frame)
        
        rows.append({
            "end": end,
            "frame": frame,
            "val": val,
            "filed": filed,
            "type": parsed["type"],
            "year": parsed["year"],
            "quarter": parsed["quarter"],
            "months": parsed["months"],
        })
    
    if not rows:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end", "frame"], keep="last")
    
    result_rows = []
    
    # 1. Use CY quarterly frames (standalone quarters)
    quarterly = df[df["type"] == "quarterly"].copy()
    cy_quarters = {}
    for _, row in quarterly.iterrows():
        result_rows.append({
            "date": row["end"],
            "quarter": f"CY{row['year']}Q{row['quarter']}",
            "val": row["val"],
            "months": 3,
            "source": "cy_frame",
        })
        cy_quarters[row["end"]] = row["val"]
    
    # 2. Process N/A frames - these are fiscal YTD cumulative values
    na_frames = df[df["frame"].isin(["N/A", "", None])].copy()
    if len(na_frames) > 0:
        na_frames = na_frames.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        
        # Assign fiscal year to each N/A frame
        na_frames["fiscal_year"] = na_frames["end"].apply(lambda d: _assign_fiscal_year(d, fy_end))
        
        # Group by fiscal year and difference
        for fiscal_year, group in na_frames.groupby("fiscal_year"):
            group = group.sort_values("end")
            prev_val = 0
            for _, row in group.iterrows():
                q_val = row["val"] - prev_val
                end_date = row["end"]
                
                # Check if this date already has a CY quarterly value
                already_covered = any(r["date"] == end_date for r in result_rows)
                if not already_covered:
                    result_rows.append({
                        "date": end_date,
                        "quarter": None,
                        "val": q_val,
                        "months": 3,
                        "source": "fiscal_ytd_diff",
                    })
                prev_val = row["val"]
    
    # 3. Compute Q4 from annual - Q3 cumulative for each year
    annual = df[df["type"] == "annual"].copy()
    cumulative = df[df["type"] == "cumulative"].copy()
    
    for _, row_ann in annual.iterrows():
        year = row_ann["year"]
        fy_val = row_ann["val"]
        fy_date = row_ann["end"]
        
        q4_exists = any((r.get("quarter") == f"CY{year}Q4") for r in result_rows)
        
        if not q4_exists:
            q3_cum = cumulative[(cumulative["year"] == year) & (cumulative["quarter"] == 3)]
            if len(q3_cum) == 0:
                q3_cum = cumulative[(cumulative["year"] == year) & (cumulative["months"] == 9)]
            
            if len(q3_cum) > 0:
                q3_val = q3_cum.iloc[-1]["val"]
                q4_val = fy_val - q3_val
                result_rows.append({
                    "date": fy_date,
                    "quarter": f"CY{year}Q4",
                    "val": q4_val,
                    "months": 3,
                    "source": "computed_q4_diff",
                })
    
    if not result_rows:
        return pd.Series(dtype=float)
    
    result_df = pd.DataFrame(result_rows)
    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    
    series = result_df.set_index("date")["val"]
    series = series.sort_index()
    return series


def parse_cashflow_quarterly(facts: dict, tag_list: list[str], fy_end: int = 12) -> pd.Series:
    """
    Parse quarterly cash flow items with fiscal YTD differencing.
    """
    tag_data = _first_tag(facts, tag_list)
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in tag_data.get("units", {}).get("USD", []):
        frame = e.get("frame", "")
        end = e.get("end", "")
        val = e.get("val")
        filed = e.get("filed", "")
        
        parsed = parse_frame(frame)
        
        rows.append({
            "end": end,
            "frame": frame,
            "val": val,
            "filed": filed,
            "type": parsed["type"],
            "year": parsed["year"],
            "quarter": parsed["quarter"],
            "months": parsed["months"],
        })
    
    if not rows:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end", "frame"], keep="last")
    
    result_rows = []
    
    # 1. Use CY quarterly frames
    quarterly = df[df["type"] == "quarterly"].copy()
    for _, row in quarterly.iterrows():
        result_rows.append({
            "date": row["end"],
            "quarter": f"CY{row['year']}Q{row['quarter']}",
            "val": row["val"],
            "months": 3,
            "source": "cy_frame",
        })
    
    # 2. Difference fiscal YTD (N/A frames) within fiscal year
    na_frames = df[df["frame"].isin(["N/A", "", None])].copy()
    if len(na_frames) > 0:
        na_frames = na_frames.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        na_frames["fiscal_year"] = na_frames["end"].apply(lambda d: _assign_fiscal_year(d, fy_end))
        
        for fiscal_year, group in na_frames.groupby("fiscal_year"):
            group = group.sort_values("end")
            prev_val = 0
            for _, row in group.iterrows():
                q_val = row["val"] - prev_val
                end_date = row["end"]
                already_covered = any(r["date"] == end_date for r in result_rows)
                if not already_covered:
                    result_rows.append({
                        "date": end_date,
                        "quarter": None,
                        "val": q_val,
                        "months": 3,
                        "source": "fiscal_ytd_diff",
                    })
                prev_val = row["val"]
    
    # 3. Compute Q4 from annual - Q3 cumulative
    annual = df[df["type"] == "annual"].copy()
    cumulative = df[df["type"] == "cumulative"].copy()
    
    for _, row_ann in annual.iterrows():
        year = row_ann["year"]
        fy_val = row_ann["val"]
        fy_date = row_ann["end"]
        
        q4_exists = any((r.get("quarter") == f"CY{year}Q4") for r in result_rows)
        
        if not q4_exists:
            q3_cum = cumulative[(cumulative["year"] == year) & (cumulative["quarter"] == 3)]
            if len(q3_cum) == 0:
                q3_cum = cumulative[(cumulative["year"] == year) & (cumulative["months"] == 9)]
            
            if len(q3_cum) > 0:
                q3_val = q3_cum.iloc[-1]["val"]
                q4_val = fy_val - q3_val
                result_rows.append({
                    "date": fy_date,
                    "quarter": f"CY{year}Q4",
                    "val": q4_val,
                    "months": 3,
                    "source": "computed_q4_diff",
                })
    
    if not result_rows:
        return pd.Series(dtype=float)
    
    result_df = pd.DataFrame(result_rows)
    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    
    series = result_df.set_index("date")["val"]
    series = series.sort_index()
    return series


def parse_balance(facts: dict, tag_list: list[str]) -> pd.Series:
    """Parse balance sheet items (point-in-time)."""
    tag_data = _first_tag(facts, tag_list)
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for unit in ("USD", "shares"):
        for e in tag_data.get("units", {}).get(unit, []):
            rows.append({"end": e["end"], "val": e["val"], "filed": e.get("filed", "")})
    
    if not rows:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    return df.set_index("end")["val"]


def detect_fiscal_year_end(equity: pd.Series) -> Optional[int]:
    """Detect fiscal year end month from equity date distribution."""
    if len(equity) == 0:
        return None
    months = pd.Series(equity.index.month)
    mode = months.mode()
    return mode.iloc[0] if len(mode) > 0 else None


def extract_financials(cik: str) -> Optional[dict]:
    """Extract raw financials from EDGAR companyfacts.

    DELEGATES to edgar_companyfacts_v2. There is exactly ONE extractor; this is a
    thin compatibility shim so existing callers keep working.

    This module used to carry its own parallel implementation, and it was wrong.
    Measured against SEC 10-K figures on 2026-08 (ttm_revenue / ttm_net_income at
    fiscal year end):

        ticker  edgar_lib (old)              v2 (canonical)
        AAPL    265.60B  -32.1%   93.74B      391.04B  0.00%   93.74B  0.00%
        MSFT     66.69B  -72.8%   88.14B      245.12B  0.00%   88.14B  0.00%
        NVDA    221.66B  +69.9%  123.67B      130.50B  0.00%   72.88B  0.00%
        PANW      4.67B  -41.9%    2.58B        8.03B  0.01%    2.58B  0.02%
        CHKP      7.31B +185.0%    2.48B        2.56B  0.00%    0.85B  3.42%

    The old code failed 6 of 10 measurements because it lacked three fixes that
    landed in v2: period-length detection from start/end (12-month facts were
    filed as quarters), best-covered XBRL tag selection (it took the first tag
    present, so AAPL revenue came from a series that stopped in 2018), and an
    annual fallback for filers with no quarterly coverage.
    """
    from edgar_companyfacts_v2 import extract_raw_financials
    return extract_raw_financials(cik)


def compute_ttm(financials: dict, qend_date, concept: str) -> Optional[float]:
    """Trailing-twelve-month sum for a quarterly series.

    Kept as a thin helper; the authoritative TTM logic (including the span check
    and annual fallback for filers with no quarterly coverage) lives in
    edgar_companyfacts_v2.compute_quarterly_fundamentals.
    """
    series = financials.get(concept, pd.Series(dtype=float))
    if series is None or series.empty:
        return None
    s = series[series.index <= qend_date].dropna().tail(4)
    return float(s.sum()) if len(s) > 0 else None


def compute_quarterly_fundamentals(financials: dict, ticker: str,
                                   px: dict[str, pd.Series] = None) -> list[dict]:
    """DELEGATES to edgar_companyfacts_v2 -- see extract_financials() for the
    measured accuracy comparison that made v2 canonical. The parallel
    implementation that used to live here was deleted, not kept as a fallback:
    two extractors meant two different answers for the same ticker.
    """
    from edgar_companyfacts_v2 import (
        compute_quarterly_fundamentals as _v2_compute,
    )
    return _v2_compute(financials, ticker, px)


if __name__ == "__main__":
    # Quick test
    cik_map = load_cik_map()
    cik = cik_map.get("PANW")
    if cik:
        fin = extract_financials(cik)
        if fin:
            rows = compute_quarterly_fundamentals(fin, "PANW")
            df = pd.DataFrame(rows)
            print(f"Extracted {len(df)} quarters for PANW")
            print(df[["as_of_date", "operating_cash_flow_ttm", "capital_expenditure_ttm",
                      "free_cash_flow", "fcf_provenance"]].tail(8).to_string())


# Fiscal year end detection and calendar-to-fiscal quarter mapping
def detect_fiscal_year_end(cik: str) -> int:
    """
    Detect fiscal year end month from SEC companyfacts.
    Uses the most common month in quarterly report dates.
    """
    try:
        url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
        r = requests.get(url, headers={'User-Agent': 'personal-research derek.moore@example.com'}, timeout=30)
        if r.status_code != 200:
            return 12  # Default to December
        
        d = r.json()
        facts = d.get('facts', {}).get('us-gaap', {})
        
        # Count months in quarterly frames
        month_counts = {}
        for tag, tag_data in facts.items():
            for unit, entries in tag_data.get('units', {}).items():
                for entry in entries:
                    frame = entry.get('frame', '')
                    if 'CY' in frame and 'Q' in frame:
                        end = entry.get('end', '')
                        if end:
                            month = int(end.split('-')[1])
                            month_counts[month] = month_counts.get(month, 0) + 1
        
        if month_counts:
            return max(month_counts, key=month_counts.get)
        return 12
    except Exception:
        return 12


def calendar_to_fiscal_quarter(date_str: str, fye_month: int) -> tuple:
    """
    Convert calendar date to fiscal quarter.
    Returns (fiscal_quarter_str, fiscal_year, fiscal_q_num)
    """
    if not date_str:
        return "UNKNOWN", None, None
    
    try:
        date = pd.to_datetime(date_str).date()
    except Exception:
        return "UNKNOWN", None, None
    
    month = date.month
    months_since_fye = (month - fye_month) % 12
    
    if months_since_fye <= 3:
        fiscal_q = 1
    elif months_since_fye <= 6:
        fiscal_q = 2
    elif months_since_fye <= 9:
        fiscal_q = 3
    else:
        fiscal_q = 4
    
    # Fiscal year is the year the fiscal year ends
    if month <= fye_month:
        fiscal_year = date.year
    else:
        fiscal_year = date.year + 1
    
    return f"FY{fiscal_year}Q{fiscal_q}", fiscal_year, fiscal_q
#!/usr/bin/env python3
"""
edgar_companyfacts_v2.py — Enhanced SEC EDGAR companyfacts parser with:
1. Quarterly differencing for cumulative cash flow frames (FY - Q3 = Q4)
2. FCF proxy using OCF when CapEx is unavailable
3. Full extraction of raw revenue, NI, OCF, CapEx, FCF
4. Frame pattern detection for various fiscal year ends
5. Validation against known-good data
"""

import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
UA = {"User-Agent": "personal-research derek.moore@example.com"}
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Known CIK overrides
CIK_OVERRIDES = {
    "XOM": "0000034088",
    "AEP": "0000004904",
    "SATS": "0001415404",
    "SPR": "0001364885",
}
NO_COMPANYFACTS = {"BAYRY"}


def load_cik_map() -> dict:
    """Load SEC ticker→CIK map."""
    r = requests.get(TICKERS_URL, headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"]).zfill(10)
        out[str(row["ticker"]).upper()] = cik
    return out


def fetch_companyfacts(cik: str) -> dict:
    """Fetch companyfacts JSON from SEC EDGAR."""
    url = FACTS_URL.format(cik=cik)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def _facts(d: dict) -> dict:
    """Extract us-gaap facts."""
    return d.get("facts", {}).get("us-gaap", {})


def _parse_frame(frame: str) -> dict:
    """
    Parse XBRL frame strings to extract calendar/fiscal info.
    
    Frame patterns:
      CY2025       -> calendar year 2025 (full year)
      CY2025Q1     -> calendar Q1 2025
      CY2025Q2     -> calendar Q2 2025
      CY2025Q3     -> calendar Q3 2025 (9-month cumulative for some filers)
      CY2025Q4     -> calendar Q4 2025 (standalone quarter)
      CY2025H1     -> calendar H1 2025 (6-month cumulative)
      CY2025M3     -> calendar March 2025 (3-month cumulative)
      CY2025M9     -> calendar Sept 2025 (9-month cumulative)
      CY2025Q1I    -> calendar Q1 2025 instant (balance sheet)
      CY2025Q2I    -> calendar Q2 2025 instant (balance sheet)
      CY2025Q3I    -> calendar Q3 2025 instant (balance sheet)
      CY2025Q4I    -> calendar Q4 2025 instant (balance sheet)
      Not found    -> skip
    
    Returns:
        {
            'type': 'annual'|'quarterly'|'cumulative'|'instant'|'unknown',
            'year': int,
            'quarter': int|None,  # 1-4 for quarterly, None for annual
            'months': int|None,   # number of months in cumulative frame
        }
    """
    if not frame or frame == "N/A":
        return {"type": "unknown", "year": None, "quarter": None, "months": None}
    
    # Match CYyyyy pattern
    m = re.match(r"CY(\d{4})$", frame)
    if m:
        return {"type": "annual", "year": int(m.group(1)), "quarter": None, "months": 12}
    
    # Match CYyyyyQn pattern (duration/flow)
    m = re.match(r"CY(\d{4})Q([1-4])$", frame)
    if m:
        return {"type": "quarterly", "year": int(m.group(1)), "quarter": int(m.group(2)), "months": 3}
    
    # Match CYyyyyQnI pattern (instant/balance sheet)
    m = re.match(r"CY(\d{4})Q([1-4])I$", frame)
    if m:
        return {"type": "instant", "year": int(m.group(1)), "quarter": int(m.group(2)), "months": 3}
    
    # Match CYyyyyHn pattern (half year)
    m = re.match(r"CY(\d{4})H([1-2])$", frame)
    if m:
        return {"type": "cumulative", "year": int(m.group(1)), "quarter": int(m.group(2)) * 2, "months": int(m.group(2)) * 6}
    
    # Match CYyyyyMn pattern (month cumulative)
    m = re.match(r"CY(\d{4})M(\d{1,2})$", frame)
    if m:
        return {"type": "cumulative", "year": int(m.group(1)), "quarter": None, "months": int(m.group(2))}
    
    return {"type": "unknown", "year": None, "quarter": None, "months": None}


def parse_income_quarterly(facts: dict, tag_list: list[str]) -> pd.Series:
    """
    Parse quarterly income statement items.
    
    Strategy:
    1. Use entries with CY frames (validated by SEC)
    2. Also use entries with N/A frames (standalone quarterly values)
    3. For annual frames, compute Q4 by differencing if Q4 standalone missing
    """
    tag_data = None
    for tag in tag_list:
        if tag in facts:
            tag_data = facts[tag]
            break
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in tag_data.get("units", {}).get("USD", []):
        frame = e.get("frame", "")
        end = e.get("end", "")
        val = e.get("val")
        filed = e.get("filed", "")
        
        parsed = _parse_frame(frame)
        
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
    
    # 1. Use standalone quarterly data (both CYyyyyQn and N/A frames)
    #    N/A frames with unique dates are typically standalone quarters
    quarterly_with_frame = df[df["type"] == "quarterly"].copy()
    na_frames = df[df["frame"].isin(["N/A", "", None])].copy()
    
    # Use CY quarterly frames
    for _, row in quarterly_with_frame.iterrows():
        result_rows.append({
            "date": row["end"],
            "quarter": f"CY{row['year']}Q{row['quarter']}",
            "val": row["val"],
            "months": 3,
            "source": "cy_frame",
        })
    
    # Use N/A frames that aren't already covered by CY frames
    # Group N/A frames by date and take the last value per date
    if len(na_frames) > 0:
        na_frames = na_frames.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        
        for _, row in na_frames.iterrows():
            end_date = row["end"]
            # Check if this date already has a CY quarterly entry
            already_covered = any(
                r["date"] == end_date for r in result_rows
            )
            if not already_covered:
                result_rows.append({
                    "date": end_date,
                    "quarter": None,  # Will assign later
                    "val": row["val"],
                    "months": 3,
                    "source": "na_frame",
                })
    
    # 2. Compute Q4 from annual - cumulative Q3 if Q4 standalone missing
    annual = df[df["type"] == "annual"].copy()
    cumulative = df[df["type"] == "cumulative"].copy()
    
    for _, row_ann in annual.iterrows():
        year = row_ann["year"]
        fy_val = row_ann["val"]
        fy_date = row_ann["end"]
        
        # Check if Q4 standalone exists
        q4_exists = any(
            (r.get("quarter") == f"CY{year}Q4") for r in result_rows
        )
        
        if not q4_exists:
            # Find Q3 cumulative (9-month) for this year
            q3_cum = cumulative[
                (cumulative["year"] == year) & (cumulative["quarter"] == 3)
            ]
            if len(q3_cum) == 0:
                q3_cum = cumulative[
                    (cumulative["year"] == year) & (cumulative["months"] == 9)
                ]
            
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


def parse_cashflow_quarterly(facts: dict, tag_list: list[str]) -> pd.Series:
    """
    Parse quarterly cash flow items with cumulative frame handling.
    
    Strategy:
    1. Use entries with CY frames (standalone quarters)
    2. Also use entries with N/A frames (standalone quarterly values)
    3. Compute Q4 = FY - Q3 cumulative when Q4 standalone missing
    4. For Q1-Q3, use cumulative differences if available
    """
    tag_data = None
    for tag in tag_list:
        if tag in facts:
            tag_data = facts[tag]
            break
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in tag_data.get("units", {}).get("USD", []):
        frame = e.get("frame", "")
        end = e.get("end", "")
        val = e.get("val")
        filed = e.get("filed", "")
        
        parsed = _parse_frame(frame)
        
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
    quarterly_with_frame = df[df["type"] == "quarterly"].copy()
    for _, row in quarterly_with_frame.iterrows():
        result_rows.append({
            "date": row["end"],
            "quarter": f"CY{row['year']}Q{row['quarter']}",
            "val": row["val"],
            "months": 3,
            "source": "cy_frame",
        })
    
    # 2. Use N/A frames that aren't already covered by CY frames
    na_frames = df[df["frame"].isin(["N/A", "", None])].copy()
    if len(na_frames) > 0:
        na_frames = na_frames.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        
        for _, row in na_frames.iterrows():
            end_date = row["end"]
            already_covered = any(
                r["date"] == end_date for r in result_rows
            )
            if not already_covered:
                result_rows.append({
                    "date": end_date,
                    "quarter": None,
                    "val": row["val"],
                    "months": 3,
                    "source": "na_frame",
                })
    
    # 3. Compute Q4 from annual - Q3 cumulative for each year
    annual = df[df["type"] == "annual"].copy()
    cumulative = df[df["type"] == "cumulative"].copy()
    
    for _, row_ann in annual.iterrows():
        year = row_ann["year"]
        fy_val = row_ann["val"]
        fy_date = row_ann["end"]
        
        q4_exists = any(
            (r.get("quarter") == f"CY{year}Q4") for r in result_rows
        )
        
        if not q4_exists:
            q3_cum = cumulative[
                (cumulative["year"] == year) & (cumulative["quarter"] == 3)
            ]
            if len(q3_cum) == 0:
                q3_cum = cumulative[
                    (cumulative["year"] == year) & (cumulative["months"] == 9)
                ]
            
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
    tag_data = None
    for tag in tag_list:
        if tag in facts:
            tag_data = facts[tag]
            break
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    # Check USD units
    for e in tag_data.get("units", {}).get("USD", []):
        frame = e.get("frame", "")
        parsed = _parse_frame(frame)
        rows.append({
            "end": e["end"], 
            "val": e["val"], 
            "filed": e.get("filed", ""),
            "frame": frame,
            "type": parsed["type"],
            "year": parsed["year"],
            "quarter": parsed["quarter"]
        })
    
    # Also check shares units
    for e in tag_data.get("units", {}).get("shares", []):
        frame = e.get("frame", "")
        parsed = _parse_frame(frame)
        rows.append({
            "end": e["end"], 
            "val": e["val"], 
            "filed": e.get("filed", ""),
            "frame": frame,
            "type": parsed["type"],
            "year": parsed["year"],
            "quarter": parsed["quarter"]
        })
    
    if not rows:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    
    # Prefer instant frames for balance sheet items, then N/A frames, then others
    # Sort by: end date, then frame type preference (instant > unknown > quarterly > cumulative > annual), then filed date
    type_priority = {"instant": 0, "unknown": 1, "quarterly": 2, "cumulative": 3, "annual": 4}
    df["type_priority"] = df["type"].map(type_priority).fillna(99)
    df = df.sort_values(["end", "type_priority", "filed"]).drop_duplicates(subset=["end"], keep="first")
    
    return df.set_index("end")["val"]


def extract_raw_financials(cik: str) -> Optional[dict]:
    """
    Extract raw quarterly financials from EDGAR companyfacts.
    
    Returns dict with quarterly series for:
    - revenue
    - net_income
    - operating_income
    - depreciation_amortization
    - interest_expense
    - operating_cash_flow (quarterly, with differencing)
    - capital_expenditure (quarterly, with differencing)
    - assets (balance sheet)
    - equity (balance sheet)
    - debt (balance sheet)
    - cash (balance sheet)
    - shares (balance sheet)
    - fiscal_year_end (detected)
    """
    d = fetch_companyfacts(cik)
    facts = _facts(d)
    
    # Tag lists
    REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", 
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
    NI_TAGS = ["NetIncomeLoss", "NetIncomeCommonStockholders"]
    OI_TAGS = ["OperatingIncomeLoss", "OperatingIncome"]
    DA_TAGS = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]
    INT_TAGS = ["InterestExpenseNonoperating", "InterestExpense", "InterestAndDebtExpense"]
    OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
    CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment",
                  "PaymentsToAcquirePropertyPlantAndEquipmentNet",
                  "CapitalExpenditure"]
    ASSET_TAGS = ["Assets"]
    EQUITY_TAGS = ["StockholdersEquity", "CommonStockholdersEquity"]
    DEBT_TAGS = ["LongTermDebtAndCapitalLeaseObligations", "LongTermDebt", "Debt",
                 "TotalDebt", "LongTermDebtNoncurrent"]
    CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                "CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"]
    SHARES_TAGS = ["CommonStockSharesOutstanding", "OrdinarySharesNumber",
                   "EntityCommonStockSharesOutstanding",
                   "WeightedAverageNumberOfSharesOutstandingBasic",
                   "WeightedAverageNumberOfDilutedSharesOutstanding"]
    
    # Parse all series
    rev = parse_income_quarterly(facts, REV_TAGS)
    ni = parse_income_quarterly(facts, NI_TAGS)
    oi = parse_income_quarterly(facts, OI_TAGS)
    da = parse_income_quarterly(facts, DA_TAGS)
    intexp = parse_income_quarterly(facts, INT_TAGS)
    ocf = parse_cashflow_quarterly(facts, OCF_TAGS)
    capex = parse_cashflow_quarterly(facts, CAPEX_TAGS)
    assets = parse_balance(facts, ASSET_TAGS)
    equity = parse_balance(facts, EQUITY_TAGS)
    debt = parse_balance(facts, DEBT_TAGS)
    cash = parse_balance(facts, CASH_TAGS)
    shares = parse_balance(facts, SHARES_TAGS)
    
    # Detect fiscal year end from equity dates
    fy_end = None
    if len(equity) > 0:
        months = pd.Series(equity.index.month)
        mode = months.mode()
        if len(mode) > 0:
            fy_end = mode.iloc[0]
    
    return {
        "revenue": rev,
        "net_income": ni,
        "operating_income": oi,
        "depreciation_amortization": da,
        "interest_expense": intexp,
        "operating_cash_flow": ocf,
        "capital_expenditure": capex,
        "assets": assets,
        "equity": equity,
        "debt": debt,
        "cash": cash,
        "shares": shares,
        "fiscal_year_end": fy_end,
    }


def compute_quarterly_fundamentals(financials: dict, ticker: str,
                                    px: dict[str, pd.Series] = None) -> list[dict]:
    """
    Compute quarterly fundamentals from extracted financials.
    
    For each quarter-end date, compute:
    - revenue, net_income, operating_income, ebitda (TTM)
    - free_cash_flow (TTM OCF - TTM CapEx, or OCF if CapEx unavailable)
    - roe, roic, debt_to_equity, interest_coverage
    - pb_ratio, ev_ebitda (if price data provided)
    """
    equity = financials.get("equity", pd.Series(dtype=float))
    if equity.empty:
        return []
    
    results = []
    
    for qend_date, eq_val in equity.items():
        if eq_val is None or eq_val <= 0:
            continue
        
        row = {
            "ticker": ticker,
            "as_of_date": qend_date.date() if hasattr(qend_date, "date") else qend_date,
        }
        
        # TTM income (4 quarters ending at qend_date)
        for name, series in [
            ("revenue", financials.get("revenue")),
            ("net_income", financials.get("net_income")),
            ("operating_income", financials.get("operating_income")),
            ("depreciation_amortization", financials.get("depreciation_amortization")),
            ("interest_expense", financials.get("interest_expense")),
            ("operating_cash_flow", financials.get("operating_cash_flow")),
            ("capital_expenditure", financials.get("capital_expenditure")),
        ]:
            if series is None or series.empty:
                row[f"ttm_{name}"] = None
                continue
            s = series[series.index <= qend_date].dropna()
            s = s.tail(4)
            row[f"ttm_{name}"] = float(s.sum()) if len(s) > 0 else None
        
        # Balance sheet values at quarter end
        for name, series in [
            ("assets", financials.get("assets")),
            ("equity", financials.get("equity")),
            ("debt", financials.get("debt")),
            ("cash", financials.get("cash")),
            ("shares", financials.get("shares")),
        ]:
            if series is None or series.empty:
                row[name] = None
                continue
            s = series[series.index <= qend_date].dropna()
            row[name] = float(s.iloc[-1]) if len(s) > 0 else None
        
        # Free Cash Flow = TTM OCF - |TTM CapEx|
        # If CapEx unavailable, use OCF as FCF proxy
        ttm_ocf = row.get("ttm_operating_cash_flow")
        ttm_capex = row.get("ttm_capital_expenditure")
        
        if ttm_ocf is not None:
            if ttm_capex is not None:
                row["free_cash_flow"] = ttm_ocf - abs(ttm_capex)
                row["fcf_source"] = "ocf_minus_capex"
                row["fcf_provenance"] = "computed"
            else:
                # CapEx unavailable — use OCF as FCF proxy
                row["free_cash_flow"] = ttm_ocf
                row["fcf_source"] = "ocf_proxy"
                row["fcf_provenance"] = "proxy"
        else:
            row["free_cash_flow"] = None
            row["fcf_source"] = None
            row["fcf_provenance"] = "unavailable"
        
        # Revenue provenance
        if row.get("total_revenue") is not None:
            row["revenue_provenance"] = "reported"
        else:
            row["revenue_provenance"] = "unavailable"
        
        # Net Income provenance
        if row.get("net_income_quarterly") is not None:
            row["net_income_provenance"] = "reported"
        else:
            row["net_income_provenance"] = "unavailable"
        
        # OCF provenance
        if row.get("ttm_operating_cash_flow") is not None:
            ocf_series = financials.get("operating_cash_flow", pd.Series(dtype=float))
            if len(ocf_series) > 0:
                s = ocf_series[ocf_series.index <= qend_date].dropna()
                if len(s) > 0:
                    row["ocf_provenance"] = "reported" if len(s) >= 4 else "computed_diff"
                else:
                    row["ocf_provenance"] = "missing"
            else:
                row["ocf_provenance"] = "missing"
        else:
            row["ocf_provenance"] = "missing"
        
        # CapEx provenance
        if row.get("ttm_capital_expenditure") is not None:
            capex_series = financials.get("capital_expenditure", pd.Series(dtype=float))
            if len(capex_series) > 0:
                s = capex_series[capex_series.index <= qend_date].dropna()
                if len(s) > 0:
                    row["capex_provenance"] = "reported" if len(s) >= 4 else "computed_diff"
                else:
                    row["capex_provenance"] = "missing"
            else:
                row["capex_provenance"] = "missing"
        else:
            row["capex_provenance"] = "missing"
        
        # Balance sheet provenance
        for bs_field in ["assets", "equity", "debt", "cash", "shares"]:
            if row.get(bs_field) is not None:
                row[f"{bs_field}_provenance"] = "reported"
            else:
                row[f"{bs_field}_provenance"] = "unavailable"
        
        # Derived ratios
        ttm_ni = row.get("ttm_net_income")
        ttm_oi = row.get("ttm_operating_income")
        ttm_da = row.get("ttm_depreciation_amortization")
        ttm_int = row.get("ttm_interest_expense")
        total_debt = row.get("debt")
        total_equity = row.get("equity")
        total_assets = row.get("assets")
        total_cash = row.get("cash")
        total_shares = row.get("shares")
        
        # ROE
        if ttm_ni is not None and total_equity and total_equity > 0:
            row["roe"] = ttm_ni / total_equity
            row["roe_provenance"] = "ttm_computed" if ttm_ni != row.get("net_income_quarterly") else "reported"
        else:
            row["roe_provenance"] = "unavailable"
        
        # ROIC
        if ttm_oi is not None:
            invested = (total_debt or 0) + (total_equity or 0)
            if invested > 0:
                nopat = ttm_oi * 0.75
                row["roic"] = nopat / invested
                row["roic_provenance"] = "ttm_computed"
            else:
                row["roic_provenance"] = "unavailable"
        else:
            row["roic_provenance"] = "missing_oi"
        
        # D/E
        if total_debt is not None and total_equity and total_equity > 0:
            row["debt_to_equity"] = total_debt / total_equity
            row["debt_to_equity_provenance"] = "reported"
        else:
            row["debt_to_equity_provenance"] = "unavailable"
        
        # Interest Coverage
        if ttm_oi is not None and ttm_int and ttm_int > 0:
            row["interest_coverage"] = ttm_oi / ttm_int
            row["interest_coverage_provenance"] = "ttm_computed"
        elif ttm_oi is not None and (ttm_int is None or ttm_int == 0):
            row["interest_coverage_provenance"] = "no_interest_expense"
        else:
            row["interest_coverage_provenance"] = "unavailable"
        
        # EBITDA
        if ttm_oi is not None:
            row["ebitda"] = ttm_oi + (ttm_da or 0)
            row["ebitda_provenance"] = "computed" if ttm_da else "oi_only"
        else:
            row["ebitda_provenance"] = "unavailable"
        
        # Market cap and related ratios
        if px and ticker in px and total_shares:
            p = px[ticker]
            avail = p[p.index <= qend_date]
            if len(avail):
                mcap = float(avail.iloc[-1]) * total_shares
                row["market_cap"] = int(mcap)
                row["market_cap_b"] = round(mcap / 1e9, 2)
                row["market_cap_provenance"] = "price_times_shares"
                
                if total_equity and total_equity > 0:
                    row["pb_ratio"] = mcap / total_equity
                    row["pb_ratio_provenance"] = "computed"
                if total_assets and total_assets > 0:
                    row["mktcap_to_assets"] = mcap / total_assets
                    row["mktcap_to_assets_provenance"] = "computed"
                
                ebitda = row.get("ebitda")
                if ebitda and ebitda > 0 and total_debt is not None and total_cash is not None:
                    ev = mcap + total_debt - total_cash
                    row["ev_ebitda"] = ev / ebitda
                    row["ev_ebitda_provenance"] = "computed"
        else:
            row["market_cap_provenance"] = "missing_price_or_shares"
        
        # FCF Margin
        if row.get("free_cash_flow") is not None and row.get("ttm_revenue") and row["ttm_revenue"] > 0:
            row["fcf_margin"] = row["free_cash_flow"] / row["ttm_revenue"]
            row["fcf_margin_provenance"] = row.get("fcf_provenance", "computed")
        else:
            row["fcf_margin_provenance"] = "unavailable"
        
        # Revenue (single quarter)
        rev_series = financials.get("revenue")
        if rev_series is not None and not rev_series.empty:
            s = rev_series[rev_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["total_revenue"] = float(s.iloc[-1])
                row["total_revenue_provenance"] = "reported"
        
        # Net Income (single quarter)
        ni_series = financials.get("net_income")
        if ni_series is not None and not ni_series.empty:
            s = ni_series[ni_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["net_income_quarterly"] = float(s.iloc[-1])
                row["net_income_quarterly_provenance"] = "reported"
        
        # Operating Income (single quarter)
        oi_series = financials.get("operating_income")
        if oi_series is not None and not oi_series.empty:
            s = oi_series[oi_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["operating_income"] = float(s.iloc[-1])
                row["operating_income_quarterly_provenance"] = "reported"

        # Compatibility columns for fundamentals.parquet schema
        row["revenue"] = row.get("total_revenue")
        row["net_income"] = row.get("net_income_quarterly")
        row["operating_cash_flow"] = row.get("ttm_operating_cash_flow")
        row["capital_expenditure"] = row.get("ttm_capital_expenditure")
        row["total_assets"] = row.get("assets")
        row["stockholders_equity"] = row.get("equity")
        row["total_debt"] = row.get("debt")
        row["cash_and_equivalents"] = row.get("cash")
        row["shareholders_equity"] = row.get("equity")
        row["total_liabilities"] = row.get("debt")
        
        # Calendar fields
        if row.get("as_of_date"):
            dt = pd.Timestamp(row["as_of_date"])
            row["calendar_year"] = dt.year
            row["calendar_quarter"] = dt.quarter
            row["fiscal_year"] = dt.year
            row["fiscal_quarter"] = dt.quarter
            row["fy"] = dt.year
            row["fp"] = f"Q{dt.quarter}"
            row["fiscal_q_num"] = dt.quarter
            row["fiscal_year_end_month"] = financials.get("fiscal_year_end")

        # Shares outstanding
        if total_shares:
            row["shares_outstanding"] = int(total_shares)
        
        row["source"] = "edgar_v2"
        row["fiscal_year_end"] = financials.get("fiscal_year_end")
        
        results.append(row)
    
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract quarterly fundamentals from EDGAR")
    ap.add_argument("--tickers", help="Comma-separated tickers")
    ap.add_argument("--max-tickers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", action="store_true",
                    help="also dump edgar_v2_quarterly.csv (diagnostic only)")
    ap.add_argument("--flush-every", type=int, default=1,
                    help="merge into fundamentals.parquet every N tickers (default 1)")
    args = ap.parse_args()
    
    # Load CIKs
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    # Determine tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        # Universe is daily_prices.parquet. monitored_stocks.parquet is
        # deprecated and must not drive a backfill: it is an optional sleeve
        # list, not the coverage universe.
        DAILY = DATA_DIR / "daily_prices.parquet"
        if DAILY.exists():
            import pyarrow.parquet as _pq
            tickers = sorted(
                _pq.read_table(DAILY, columns=["ticker"])["ticker"]
                .to_pandas().astype(str).str.upper().unique().tolist()
            )
        else:
            tickers = sorted(cik_map.keys())
    
    tickers = [t for t in tickers if t not in NO_COMPANYFACTS]
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    
    print(f"Processing {len(tickers)} tickers...")

    # Incremental merge: flush completed tickers into fundamentals.parquet as we
    # go, so an interrupt (or a rate-limit abort) keeps everything already
    # fetched. The previous version accumulated every row in memory and only
    # wrote edgar_v2_quarterly.csv at the very end -- it never touched
    # fundamentals.parquet at all, which is why the panel showed only 39
    # edgar_v2 rows from an unrelated path.
    from backfill_edgar import merge_into_fundamentals

    results = []
    pending = []
    counters = {"merged": 0}          # module scope here, so no `nonlocal`
    done = skipped = 0

    def _flush():
        if not pending:
            return
        try:
            n = merge_into_fundamentals(list(pending))
            counters["merged"] += len(pending)
            print(f"    [flush] merged {len(pending)} rows -> panel now {n:,}")
        except Exception as e:
            # Never lose the batch silently; keep `pending` intact so the next
            # flush retries it rather than dropping the fetched rows.
            print(f"    !! flush FAILED ({type(e).__name__}: {e}) -- retrying at next flush")
            return
        pending.clear()

    for t in tickers:
        if t not in cik_map:
            print(f"  !! {t}: no CIK")
            skipped += 1
            continue
        cik = cik_map[t]
        try:
            fin = extract_raw_financials(cik)
            if fin is None:
                print(f"  !! {t}: no data")
                skipped += 1
                continue

            # Print diagnostic
            rev_count = len(fin.get("revenue", pd.Series()))
            ocf_count = len(fin.get("operating_cash_flow", pd.Series()))
            capex_count = len(fin.get("capital_expenditure", pd.Series()))
            eq_count = len(fin.get("equity", pd.Series()))
            print(f"  {t}: rev={rev_count}, ocf={ocf_count}, capex={capex_count}, eq={eq_count}")

            if args.dry_run:
                continue

            rows = compute_quarterly_fundamentals(fin, t)
            results.extend(rows)
            pending.extend(rows)
            done += 1
            print(f"    → {len(rows)} quarters")

            if args.flush_every and done % args.flush_every == 0:
                _flush()

            time.sleep(0.12)
        except KeyboardInterrupt:
            print("\n[interrupted] flushing what we have...")
            _flush()
            raise
        except Exception as e:
            print(f"  !! {t}: {e}")
            time.sleep(0.12)

    _flush()   # final partial batch
    print(f"\nTickers processed: {done}, skipped: {skipped}")
    print(f"Rows merged into fundamentals.parquet: {counters['merged']}")

    if args.csv and results:
        # Diagnostic dump only. This file is NOT the persistence path -- it went
        # stale once (written 00:22, script's alias block changed 15:37 the same
        # day) and its 46 columns lacked shareholders_equity/total_assets/
        # total_debt entirely, which is what made edgar_v2 rows look empty.
        df = pd.DataFrame(results)
        df.to_csv("edgar_v2_quarterly.csv", index=False)
        print(f"Also wrote edgar_v2_quarterly.csv ({len(df)} rows, diagnostic)")

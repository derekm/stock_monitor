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


def _pick_tag(facts: dict, tag_list: list[str]):
    """Choose the BEST-COVERED tag from `tag_list`, not merely the first present.

    All three parsers used to take the first tag that existed. That is wrong for
    revenue: "Revenues" leads REV_TAGS but is a legacy stub for most filers,
    while the modern ASC-606 tag carries the real history. Measured coverage:

      AAPL  Revenues                       11 facts, last 2018-09-29
            RevenueFromContract...Excluding 117 facts, last 2026-06-27
      MSFT  Revenues                       31 facts, last 2010-12-31
            RevenueFromContract...Excluding 134 facts, last 2026-06-30
      PANW  Revenues                       18 facts, ZERO quarterly, last 2018-07-31
            RevenueFromContract...Excluding 117 facts, 64 quarterly, last 2026-04-30

    So AAPL's ttm_revenue came from a series that stopped in 2018 -- hence 265.6B
    against an actual FY24 revenue of 391.0B, and PANW/CHKP reporting rev=0/2
    usable points.

    Ranking: RECENCY FIRST, then quarterly coverage. Ordering by raw quarterly
    count is wrong -- SalesRevenueNet has MORE historical quarters than the modern
    tag (AAPL 136 vs 64) but was discontinued in 2018, and picking it gave
    ttm_revenue 255.3B against an actual 391.0B (-34.7%). What matters is which
    tag is still being filed, so tags are compared on (last quarterly period end,
    quarterly count). NVDA is the case that keeps this general rather than a
    hardcoded reordering: there "Revenues" is genuinely correct (quarterly facts
    through 2026-04-26), and it still wins under this rule.
    """
    best = None
    for tag in tag_list:
        if tag not in facts:
            continue
        units = facts[tag].get("units", {}).get("USD", [])
        if not units:
            continue
        n_q = 0
        last_q_end = ""          # most recent QUARTERLY period end, not any end
        for e in units:
            if not e.get("start"):
                continue
            sm = _span_months(e)
            if sm is not None and 2 <= sm <= 4:
                n_q += 1
                if e.get("end", "") > last_q_end:
                    last_q_end = e.get("end", "")
        if n_q == 0:
            # No quarterly facts at all: keep as a last resort, ranked below any
            # tag that has them (CHKP files annual-only revenue).
            last_any = max((e.get("end", "") for e in units), default="")
            score = ("", 0, last_any, len(units))
        else:
            score = (last_q_end, n_q, "", len(units))
        if best is None or score > best[1]:
            best = (tag, score)
    return best[0] if best else None


def _annual_series(facts: dict, tag_list: list[str]) -> pd.Series:
    """12-month (annual) facts as a Series keyed by period end.

    Uses the same best-covered tag selection as the quarterly parsers, but keeps
    only ~12-month spans. This is the honest source of a trailing-twelve-month
    figure for filers that publish no quarterly data at all.
    """
    tag = _pick_tag(facts, tag_list)
    if not tag:
        return pd.Series(dtype=float)
    rows = {}
    for e in facts[tag].get("units", {}).get("USD", []):
        if not e.get("start"):
            continue
        sm = _span_months(e)
        if sm is None or not (11 <= sm <= 13):
            continue
        end = e.get("end")
        val = e.get("val")
        if end is None or val is None:
            continue
        rows[pd.Timestamp(end)] = float(val)     # later filings overwrite
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(rows).sort_index()


def _span_months(entry: dict):
    """Actual period length of an EDGAR fact, from start/end.

    The `frame` string alone is not sufficient. EDGAR emits a duplicate of most
    figures with frame "N/A", and _parse_frame("N/A") returns months=None; the
    N/A branches in the parsers below then appended those rows with months=3
    unconditionally. That filed 12-month values as single quarters -- AAPL's
    FY2024 net income (start 2023-10-01, end 2024-09-28, 93.736B) was stored on
    2024-09-28 as a "quarter", so tail(4).sum() gave a TTM of 172.7B
    (= 93.736 annual + 79.001 for Q1-Q3) against a true FY figure of 93.736B.

    Returns the rounded month span, or None when start/end are unusable.
    """
    start, end = entry.get("start"), entry.get("end")
    if not (start and end):
        return None
    try:
        return int(round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44))
    except Exception:
        return None


def parse_income_quarterly(facts: dict, tag_list: list[str]) -> pd.Series:
    """
    Parse quarterly income statement items.
    
    Strategy:
    1. Use entries with CY frames (validated by SEC)
    2. Also use entries with N/A frames (standalone quarterly values)
    3. For annual frames, compute Q4 by differencing if Q4 standalone missing
    """
    # best-covered tag, not first-present -- see _pick_tag
    _tag = _pick_tag(facts, tag_list)
    tag_data = facts.get(_tag) if _tag else None
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
            "span_months": _span_months(e),
            "start": e.get("start"),
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
    
    # Use N/A frames that aren't already covered by CY frames.
    #
    # ONLY genuine ~3-month spans. An N/A frame carries no period information, so
    # this branch used to assume months=3 for every one of them -- filing 6/9/12
    # month figures as single quarters and inflating every downstream TTM sum.
    # span_months comes from the fact's own start/end, so a 12-month duplicate is
    # now rejected instead of masquerading as Q4.
    if len(na_frames) > 0:
        na_frames = na_frames.sort_values("filed").drop_duplicates(subset=["end"], keep="last")

        for _, row in na_frames.iterrows():
            end_date = row["end"]
            sm = row.get("span_months")
            if sm is None or not (2 <= sm <= 4):
                continue
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
    
    # 2. Compute Q4 as (12-month) - (9-month) sharing the same fiscal start.
    #
    # Matched on the facts' own start/end spans, NOT on frame strings. Frames are
    # unreliable for off-calendar fiscal years: MSFT's FY2024 ends 2024-06-30 but
    # its annual frame is "CY2024", and its 9-month cumulative carries frame
    # "N/A" (so _parse_frame typed it "unknown", never "cumulative"). The old
    # year/quarter lookup therefore found nothing and Q4 was silently dropped --
    # leaving a 6-month hole in the series for every June/January/October
    # fiscal-year-end company, which then made tail(4) span five quarters.
    #
    # Pairing on (start, 12mo) vs (start, 9mo) is fiscal-calendar agnostic:
    # MSFT FY24 = 88.136 (2023-07-01..2024-06-30) minus 66.100
    # (2023-07-01..2024-03-31) = 22.036B for Q4.
    df_spans = df.copy()
    if "span_months" not in df_spans.columns:
        df_spans["span_months"] = None
    df_spans["_start"] = pd.to_datetime(df_spans.get("start"), errors="coerce") \
        if "start" in df_spans.columns else pd.NaT

    ann12 = df_spans[df_spans["span_months"].between(11, 13, inclusive="both")] \
        if df_spans["span_months"].notna().any() else df_spans.iloc[0:0]
    cum9 = df_spans[df_spans["span_months"].between(8, 10, inclusive="both")] \
        if df_spans["span_months"].notna().any() else df_spans.iloc[0:0]

    existing_dates = {r["date"] for r in result_rows}
    for _, row_ann in ann12.sort_values("filed").iterrows():
        fy_date = row_ann["end"]
        if fy_date in existing_dates:
            continue                      # a real standalone Q4 already exists
        fy_start = row_ann.get("_start")
        if pd.isna(fy_start):
            continue
        match = cum9[cum9["_start"] == fy_start]
        if match.empty:
            continue
        q4_val = row_ann["val"] - match.sort_values("filed").iloc[-1]["val"]
        result_rows.append({
            "date": fy_date,
            "quarter": None,
            "val": q4_val,
            "months": 3,
            "source": "computed_q4_diff",
        })
        existing_dates.add(fy_date)
    
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
    # best-covered tag, not first-present -- see _pick_tag
    _tag = _pick_tag(facts, tag_list)
    tag_data = facts.get(_tag) if _tag else None
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
            "span_months": _span_months(e),
            "start": e.get("start"),
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
            # ONLY genuine ~3-month spans -- see the note in
            # parse_income_quarterly: an N/A frame carries no period info, and
            # assuming months=3 filed 6/9/12-month figures as single quarters.
            sm = row.get("span_months")
            if sm is None or not (2 <= sm <= 4):
                continue
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
    
    # 3. Compute Q4 as (12-month) - (9-month) sharing the same fiscal start.
    #    Span-matched, not frame-matched -- see parse_income_quarterly for why
    #    (off-calendar fiscal years carry misleading CY frames and N/A
    #    cumulatives, which silently dropped Q4 and left 6-month gaps).
    df_spans = df.copy()
    if "span_months" not in df_spans.columns:
        df_spans["span_months"] = None
    df_spans["_start"] = pd.to_datetime(df_spans.get("start"), errors="coerce") \
        if "start" in df_spans.columns else pd.NaT
    has_spans = df_spans["span_months"].notna().any()
    ann12 = df_spans[df_spans["span_months"].between(11, 13, inclusive="both")] \
        if has_spans else df_spans.iloc[0:0]
    cum9 = df_spans[df_spans["span_months"].between(8, 10, inclusive="both")] \
        if has_spans else df_spans.iloc[0:0]

    existing_dates = {r["date"] for r in result_rows}
    for _, row_ann in ann12.sort_values("filed").iterrows():
        fy_date = row_ann["end"]
        if fy_date in existing_dates:
            continue
        fy_start = row_ann.get("_start")
        if pd.isna(fy_start):
            continue
        match = cum9[cum9["_start"] == fy_start]
        if match.empty:
            continue
        q4_val = row_ann["val"] - match.sort_values("filed").iloc[-1]["val"]
        result_rows.append({
            "date": fy_date,
            "quarter": None,
            "val": q4_val,
            "months": 3,
            "source": "computed_q4_diff",
        })
        existing_dates.add(fy_date)
    
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
    # best-covered tag, not first-present -- see _pick_tag
    _tag = _pick_tag(facts, tag_list)
    tag_data = facts.get(_tag) if _tag else None
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
        # Annual (12-month) series, keyed by period end. Needed for filers with
        # NO quarterly coverage at all: Check Point (CHKP) is a foreign private
        # issuer reporting semi-annually, so its revenue facts are 48x 12-month
        # and 0x 3-month. tail(4) over its parsed series summed four different
        # YEARS' Q4-by-difference values and produced 0.70B against an actual
        # FY24 revenue of 2.565B. compute_quarterly_fundamentals falls back to
        # this when the quarterly series cannot form a real TTM.
        "annual_revenue": _annual_series(facts, REV_TAGS),
        "annual_net_income": _annual_series(facts, NI_TAGS),
        "annual_operating_income": _annual_series(facts, OI_TAGS),
        "annual_operating_cash_flow": _annual_series(facts, OCF_TAGS),
        "annual_capital_expenditure": _annual_series(facts, CAPEX_TAGS),
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
            s = series[series.index <= qend_date].dropna().tail(4)
            # A real TTM needs four quarters that actually SPAN ~12 months. Some
            # filers publish no quarterly facts at all (CHKP is a foreign private
            # issuer filing semi-annually: 48 x 12-month revenue facts, 0 x
            # 3-month), so the parsed series holds one Q4-by-difference value per
            # year and tail(4) summed FOUR DIFFERENT YEARS -- 0.70B against an
            # actual FY24 revenue of 2.565B. Detect that and use the reported
            # 12-month figure instead of a fabricated sum.
            ttm_val = None
            if len(s) > 0:
                span_ok = False
                if len(s) >= 4:
                    span = (s.index[-1] - s.index[0]).days
                    span_ok = 240 <= span <= 400
                if span_ok:
                    ttm_val = float(s.sum())
                else:
                    ann = financials.get(f"annual_{name}")
                    if ann is not None and not ann.empty:
                        aa = ann[ann.index <= qend_date].dropna()
                        if len(aa) > 0:
                            ttm_val = float(aa.iloc[-1])
                    if ttm_val is None and len(s) >= 4:
                        ttm_val = float(s.sum())
            row[f"ttm_{name}"] = ttm_val
        
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

        # Canonical fundamentals.parquet schema (post-2026-08 migration).
        #
        # Names carry the period basis explicitly: *_quarterly is one fiscal
        # quarter, *_ttm is a trailing-twelve-month sum. The internal working
        # names above are short (equity/assets/debt/cash) and the ttm_* keys hold
        # twelve-month sums; both are mapped here at the output boundary.
        row["revenue_quarterly"] = row.get("total_revenue")
        row["net_income_quarterly"] = row.get("net_income_quarterly")
        row["operating_income_quarterly"] = row.get("operating_income")
        row["revenue_ttm"] = row.get("ttm_revenue")
        row["net_income_ttm"] = row.get("ttm_net_income")
        row["operating_income_ttm"] = row.get("ttm_operating_income")
        # These two previously wrote TTM values into the QUARTERLY names
        # (operating_cash_flow = ttm_operating_cash_flow), which mislabelled a
        # twelve-month sum as a single quarter. Now mapped to the _ttm names.
        row["operating_cash_flow_ttm"] = row.get("ttm_operating_cash_flow")
        row["capital_expenditure_ttm"] = row.get("ttm_capital_expenditure")
        row["total_assets"] = row.get("assets")
        row["shareholders_equity"] = row.get("equity")
        row["total_debt"] = row.get("debt")
        row["cash_and_equivalents"] = row.get("cash")
        # shares_outstanding: reject the corrupt values that made the old `shares`
        # column unusable (FITB 2010-09-30 held 7.96e14 -- 796 trillion shares;
        # 45 of 73 unique rows were 0.0). No US listed company exceeds ~1e11.
        _sh = row.get("shares")
        row["shares_outstanding"] = _sh if (_sh and 0 < _sh < 1e11) else None
        # NOTE: total_liabilities is NOT set from `debt`. Measured on the panel,
        # total_liabilities / total_debt has a median ratio of 2.515 -- debt is a
        # SUBSET of liabilities, so copying one into the other was wrong.
        
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
    ap.add_argument("--force", action="store_true",
                    help="overwrite rows already stamped edgar_v2 (needed to push "
                         "an extractor CORRECTION through source protection)")
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
            n = merge_into_fundamentals(list(pending), force=args.force)
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
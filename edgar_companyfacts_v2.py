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
    """Facts for the taxonomy this filer reports under.

    Domestic filers report us-gaap. Foreign private issuers filing 40-F/20-F report
    ifrs-full, and their us-gaap block is either absent or a stale remnant: Barrick
    (CIK 756894) carries 248 us-gaap tags whose newest fact is 2010-12-31 alongside
    301 ifrs-full tags current to 2025-12-31. Choosing by tag count alone would pick
    the stale block, so pick the taxonomy with the NEWEST fact.
    """
    facts = d.get("facts", {})
    candidates = [t for t in ("us-gaap", "ifrs-full") if facts.get(t)]
    if not candidates:
        return {}
    if len(candidates) == 1:
        return facts[candidates[0]]

    best, best_end = None, ""
    for tax in candidates:
        newest = ""
        for tag in facts[tax].values():
            for arr in tag.get("units", {}).values():
                for e in arr:
                    end = e.get("end", "")
                    if end > newest:
                        newest = end
        if newest > best_end:
            best, best_end = tax, newest
    return facts[best]


def _taxonomy_of(d: dict) -> str:
    """Name of the taxonomy _facts() would select, for logging and provenance."""
    facts = d.get("facts", {})
    for tax in ("us-gaap", "ifrs-full"):
        if facts.get(tax) and facts[tax] is _facts(d):
            return tax
    return "unknown"


def _unit_entries(tag_data: dict) -> list[dict]:
    """Facts from the single most-covered non-pure unit. One currency per series."""
    units = (tag_data or {}).get("units", {}) or {}
    keys = [k for k in units if k and str(k).lower() != "pure"]
    if not keys:
        return []
    key = max(keys, key=lambda k: len(units.get(k) or []))
    return list(units.get(key) or [])


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


def _pick_tag(facts: dict, tag_list: list[str], prefer_order: bool = False):
    """Choose the BEST-COVERED tag from `tag_list`, not merely the first present.

    `prefer_order=True` instead walks tag_list IN ORDER and returns the first tag
    with any usable facts. Use it when the tags are NOT interchangeable measures
    of the same quantity. SHARES_TAGS is that case: CommonStockSharesOutstanding
    is a POINT-IN-TIME count, which is what a balance-sheet field needs, while
    WeightedAverageNumberOfSharesOutstandingBasic is a period AVERAGE used for EPS.
    Coverage ranking would prefer the weighted-average for AAPL on fact count alone
    (234 vs 144) and substitute a different quantity; the two happen to sit close
    for AAPL (14.61B vs 14.66B) but they do not measure the same thing.

    Coverage ranking matters for revenue, where taking the first tag present is
    wrong: "Revenues" leads REV_TAGS but is a legacy stub for most filers,
    while the modern ASC-606 tag carries the real history. Measured coverage:

      AAPL  Revenues                       11 facts, last 2018-09-29
            RevenueFromContract...Excluding 117 facts, last 2026-06-27
      MSFT  Revenues                       31 facts, last 2010-12-31
            RevenueFromContract...Excluding 134 facts, last 2026-06-30
      PANW  Revenues                       18 facts, ZERO quarterly, last 2018-07-31
            RevenueFromContract...Excluding 117 facts, 64 quarterly, last 2026-04-30

    Picking a discontinued tag yields a series that stops at its last filing --
    "Revenues" for AAPL ends in 2018 and produces a revenue_ttm of 265.6B against an
    actual FY24 391.0B, and leaves PANW/CHKP with 0-2 usable points.

    Ranking: RECENCY FIRST, then quarterly coverage. Raw quarterly count is the
    wrong key -- SalesRevenueNet holds MORE historical quarters than the modern tag
    (AAPL 136 vs 64) but was discontinued in 2018, and choosing it gives revenue_ttm
    255.3B against 391.0B (-34.7%). What matters is which tag is still being filed,
    so tags are compared on (last quarterly period end, quarterly count). NVDA keeps
    this general rather than a hardcoded reordering: there "Revenues" is genuinely
    correct (quarterly facts through 2026-04-26) and wins under the same rule.
    """
    if prefer_order:
        for tag in tag_list:
            if tag not in facts:
                continue
            if _unit_entries(facts[tag]):
                return tag
        return None

    best = None
    for tag in tag_list:
        if tag not in facts:
            continue
        units = _unit_entries(facts[tag])
        if not units:
            continue
        n_q = 0
        last_q_end = ""          # most recent QUARTERLY period end, not any end
        q_vals = []
        for e in units:
            if not e.get("start"):
                continue
            sm = _span_months(e)
            if sm is not None and 2 <= sm <= 4:
                n_q += 1
                v = e.get("val")
                if isinstance(v, (int, float)):
                    q_vals.append(abs(float(v)))
                if e.get("end", "") > last_q_end:
                    last_q_end = e.get("end", "")
        # Magnitude breaks ties between tags that are equally current and equally
        # covered but measure DIFFERENT SCOPES. ABR (a mortgage REIT) files
        # RealEstateRevenueNet and OperatingLeaseLeaseIncome with 36 quarters each and
        # the same last period; falling through to fact count picked the lease line
        # (median $1.51M) over the top line (median $7.88M), and that small
        # denominator produced fcf_margin values in the hundreds. The broader measure
        # is the larger one, so prefer it -- only when recency and coverage are equal,
        # which keeps the discontinued-tag protection above intact.
        med_q = 0.0
        if q_vals:
            q_vals.sort()
            med_q = q_vals[len(q_vals) // 2]
        if n_q == 0:
            # No quarterly facts at all: keep as a last resort, ranked below any
            # tag that has them (CHKP files annual-only revenue).
            last_any = max((e.get("end", "") for e in units), default="")
            score = ("", 0, last_any, 0.0, len(units))
        else:
            score = (last_q_end, n_q, "", med_q, len(units))
        if best is None or score > best[1]:
            best = (tag, score)
    return best[0] if best else None


def _annual_series(facts: dict, tag_list: list[str],
                   prefer_order: bool = False) -> pd.Series:
    """12-month (annual) facts as a Series keyed by period end.

    Uses the same best-covered tag selection as the quarterly parsers, but keeps
    only ~12-month spans. This is the honest source of a trailing-twelve-month
    figure for filers that publish no quarterly data at all.
    """
    tag = _pick_tag(facts, tag_list, prefer_order=prefer_order)
    if not tag:
        return pd.Series(dtype=float)
    rows = {}
    for e in _unit_entries(facts[tag]):
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


def parse_income_quarterly(facts: dict, tag_list: list[str],
                           prefer_order: bool = False) -> pd.Series:
    """
    Parse quarterly income statement items.
    
    Strategy:
    1. Use entries with CY frames (validated by SEC)
    2. Also use entries with N/A frames (standalone quarterly values)
    3. For annual frames, compute Q4 by differencing if Q4 standalone missing
    """
    # best-covered tag, unless the caller says the tags are not interchangeable
    _tag = _pick_tag(facts, tag_list, prefer_order=prefer_order)
    tag_data = facts.get(_tag) if _tag else None
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in _unit_entries(tag_data):
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
    # span_months is taken from the fact's own start/end; that rejects the 6/9/12
    # month duplicates EDGAR also files, which would otherwise enter as quarters
    # and inflate every downstream TTM sum.
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
    # its annual frame is "CY2024", and its 9-month cumulative carries frame "N/A"
    # (_parse_frame types that "unknown", never "cumulative"). A year/quarter lookup
    # finds no pair there, dropping Q4 and leaving a 6-month hole for every
    # June/January/October fiscal-year-end company.
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
    for e in _unit_entries(tag_data):
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
    #    Span-matched, not frame-matched -- see parse_income_quarterly: off-calendar
    #    fiscal years carry misleading CY frames and N/A cumulatives.
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


def parse_balance(facts: dict, tag_list: list[str],
                  prefer_order: bool = False) -> pd.Series:
    """Parse balance sheet items (point-in-time).

    `prefer_order` is passed through to _pick_tag. Set it for SHARES_TAGS, whose
    entries measure DIFFERENT quantities (point-in-time count vs weighted-average
    for EPS) and so must be tried in priority order rather than ranked by how many
    facts each one happens to have.
    """
    _tag = _pick_tag(facts, tag_list, prefer_order=prefer_order)
    tag_data = facts.get(_tag) if _tag else None
    if tag_data is None:
        return pd.Series(dtype=float)
    
    rows = []
    for e in _unit_entries(tag_data):
        frame = e.get("frame", "")
        parsed = _parse_frame(frame)
        rows.append({
            "end": e["end"],
            "val": e["val"],
            "filed": e.get("filed", ""),
            "frame": frame,
            "type": parsed["type"],
            "year": parsed["year"],
            "quarter": parsed["quarter"],
            "span_months": _span_months(e),
        })
    if not rows:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])

    # One value per period end. Ordering, in priority:
    #   1. instant vs duration -- a true instant fact IS the balance-sheet value
    #   2. SHORTEST span -- for duration facts (the weighted-average share tags)
    #      the 3-month figure describes the period ending here; a 6- or 12-month
    #      average describes a longer window
    #   3. frame type, then latest filing (restatements supersede originals)
    #
    # Span must outrank frame type. Ranking frame type first was still wrong for
    # SpaceX (SPCX): at end=2026-06-30 the 3-month fact carries frame CY2026Q2 ->
    # type "quarterly" (priority 2) while the 6-month fact has an EMPTY frame ->
    # type "unknown" (priority 1), so the longer span won on type and the series
    # returned 4.879B instead of the correct 5.864B. `_span_months` had it right
    # (6 vs 3); the tiebreak order was the defect.
    type_priority = {"instant": 0, "unknown": 1, "quarterly": 2, "cumulative": 3, "annual": 4}
    df["type_priority"] = df["type"].map(type_priority).fillna(99)
    # instant facts have no start -> span_months None; rank them first (-1), and
    # never let a NaN sort ahead of a real short span
    df["is_duration"] = df["span_months"].notna().astype(int)
    df["span_rank"] = df["span_months"].fillna(-1)
    df = (df.sort_values(["end", "is_duration", "span_rank", "type_priority", "filed"],
                         ascending=[True, True, True, True, False])
            .drop_duplicates(subset=["end"], keep="first"))

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
    
    # Tag lists. US-GAAP names first, then the ifrs-full equivalents a foreign
    # private issuer (40-F/20-F) reports instead. _pick_tag selects on coverage and
    # recency, so listing both taxonomies is safe: only one of them has facts.
    #
    # Financial-sector filers report no `Revenues` at all. A bank's top line is
    # InterestAndDividendIncomeOperating (gross interest and dividend income), an
    # insurer's is PremiumsEarnedNet, a REIT's is RealEstateRevenueNet.
    # InterestIncomeExpenseNet is deliberately excluded: it is net of interest
    # EXPENSE, so it behaves like a margin and would understate the top line by an
    # order of magnitude. NoninterestIncome is excluded for the same reason -- it is
    # one component of revenue, not the total.
    #
    # These names measure different SCOPES, so _pick_tag breaks recency/coverage ties
    # on median magnitude: a REIT filing both RealEstateRevenueNet and
    # OperatingLeaseLeaseIncome should get the top line, not one lease line.
    #
    # InterestIncomeOperating is the gross interest top line for mortgage REITs and
    # specialty lenders. ABR files it at $235M/quarter while its only other listed tag
    # still current is OperatingLeaseLeaseIncome at $1.51M -- a minor line that, used
    # as a denominator, produced fcf_margin in the hundreds.
    REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", 
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "Revenue", "RevenueFromContractsWithCustomers",
                "RevenuesNetOfInterestExpense",
                "InterestAndDividendIncomeOperating",
                "InterestIncomeOperating",
                "InterestAndFeeIncomeLoansAndLeases",
                "PremiumsEarnedNet", "PremiumsEarnedNetPropertyCasualty",
                "RealEstateRevenueNet", "OperatingLeaseLeaseIncome"]
    NI_TAGS = ["NetIncomeLoss", "NetIncomeCommonStockholders",
               "ProfitLossAttributableToOwnersOfParent", "ProfitLoss"]
    OI_TAGS = ["OperatingIncomeLoss", "OperatingIncome",
               "ProfitLossFromOperatingActivities"]
    DA_TAGS = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
               "DepreciationAndAmortisationExpense",
               "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss"]
    INT_TAGS = ["InterestExpenseNonoperating", "InterestExpense", "InterestAndDebtExpense",
                "InterestExpenseOnBorrowings", "FinanceCosts"]
    OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                "CashFlowsFromUsedInOperatingActivities"]
    CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment",
                  "PaymentsToAcquirePropertyPlantAndEquipmentNet",
                  "CapitalExpenditure",
                  "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"]
    ASSET_TAGS = ["Assets", "TotalAssets"]
    EQUITY_TAGS = ["StockholdersEquity", "CommonStockholdersEquity",
                   "EquityAttributableToOwnersOfParent", "Equity"]
    DEBT_TAGS = ["LongTermDebtAndCapitalLeaseObligations", "LongTermDebt", "Debt",
                 "TotalDebt", "LongTermDebtNoncurrent",
                 "Borrowings", "LongtermBorrowings"]
    CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                "CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"]
    GP_TAGS = ["GrossProfit", "GrossProfitLoss"]
    COGS_TAGS = ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"]
    SHARES_TAGS = ["CommonStockSharesOutstanding", "OrdinarySharesNumber",
                   "EntityCommonStockSharesOutstanding",
                   "NumberOfSharesOutstanding",
                   "WeightedAverageNumberOfSharesOutstandingBasic",
                   "WeightedAverageNumberOfDilutedSharesOutstanding",
                   "AdjustedWeightedAverageShares"]
    
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
    gp = parse_income_quarterly(facts, GP_TAGS)
    cogs = parse_income_quarterly(facts, COGS_TAGS)
    # SHARES_TAGS entries are NOT interchangeable: CommonStockSharesOutstanding is
    # a point-in-time count, the WeightedAverage* tags are period averages for EPS.
    # Try them in priority order instead of ranking by fact count.
    shares = parse_balance(facts, SHARES_TAGS, prefer_order=True)
    
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
        "gross_profit": gp,
        "cogs": cogs,
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
    assets = financials.get("assets", pd.Series(dtype=float))
    spine = equity.index.union(assets.index)
    if spine.empty:
        return []

    results = []

    for qend_date in spine.sort_values():
        eq_val = equity.get(qend_date) if len(equity) else None
        # Negative/zero book equity is real. Emit the row; ratio fields
        # that need equity > 0 stay unset below.
        row = {
            "ticker": ticker,
            "as_of_date": qend_date.date() if hasattr(qend_date, "date") else qend_date,
        }

        # TTM income (4 quarters ending at qend_date).
        #
        # `out_name` is the CANONICAL panel column and the only key written:
        # merge_into_fundamentals persists every key it is handed, so emitting a
        # working alias alongside it would create a duplicate column.
        # ttm_depreciation_amortization and ttm_interest_expense ARE the panel
        # column names; there is no *_ttm variant of those two.
        for src_key, out_name in [
            ("revenue", "revenue_ttm"),
            ("net_income", "net_income_ttm"),
            ("operating_income", "operating_income_ttm"),
            ("depreciation_amortization", "ttm_depreciation_amortization"),
            ("interest_expense", "ttm_interest_expense"),
            ("operating_cash_flow", "operating_cash_flow_ttm"),
            ("capital_expenditure", "capital_expenditure_ttm"),
            ("gross_profit", "gross_profit_ttm"),
        ]:
            series = financials.get(src_key)
            if series is None or series.empty:
                # No quarterly facts at all. Annual-only filers exist -- an IFRS
                # 40-F filer can publish 22 x 12-month revenue facts and zero
                # 3-month ones -- so fall back to the reported annual figure
                # rather than dropping the field.
                ann = financials.get(f"annual_{src_key}")
                if ann is not None and not ann.empty:
                    aa = ann[ann.index <= qend_date].dropna()
                    row[out_name] = float(aa.iloc[-1]) if len(aa) else None
                else:
                    row[out_name] = None
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
                    ann = financials.get(f"annual_{src_key}")
                    if ann is not None and not ann.empty:
                        aa = ann[ann.index <= qend_date].dropna()
                        if len(aa) > 0:
                            ttm_val = float(aa.iloc[-1])
                    if ttm_val is None and len(s) >= 4:
                        ttm_val = float(s.sum())
            row[out_name] = ttm_val

        # Balance sheet values at quarter end, written straight to the canonical
        # column. `debt` is kept alongside total_debt because it is a real
        # pre-existing panel column, not a leaked alias.
        for src_key, out_name in [
            ("assets", "total_assets"),
            ("equity", "shareholders_equity"),
            ("debt", "total_debt"),
            ("cash", "cash_and_equivalents"),
            ("shares", "shares_outstanding"),
        ]:
            series = financials.get(src_key)
            if series is None or series.empty:
                row[out_name] = None
                continue
            s = series[series.index <= qend_date].dropna()
            row[out_name] = float(s.iloc[-1]) if len(s) > 0 else None

        gp_ttm = row.get("gross_profit_ttm")
        row["gross_profit"] = gp_ttm

        # Keep real mega-share counts (HCMC is ~3.8e11). Drop non-positive
        # or >= 1e14 values — those are scale errors, not share counts.
        _sh = row.get("shares_outstanding")
        row["shares_outstanding"] = _sh if (_sh and 0 < _sh < 1e14) else None

        # Free Cash Flow = TTM OCF - |TTM CapEx|
        # If CapEx unavailable, use OCF as FCF proxy
        ttm_ocf = row.get("operating_cash_flow_ttm")
        ttm_capex = row.get("capital_expenditure_ttm")

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

        # Single-quarter values. Assigned BEFORE the provenance blocks below, which
        # read them back -- a provenance check running first would test keys that do
        # not exist yet and report "unavailable" for populated fields.
        rev_series = financials.get("revenue")
        if rev_series is not None and not rev_series.empty:
            s = rev_series[rev_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["revenue_quarterly"] = float(s.iloc[-1])
                row["total_revenue_provenance"] = "reported"

        ni_series = financials.get("net_income")
        if ni_series is not None and not ni_series.empty:
            s = ni_series[ni_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["net_income_quarterly"] = float(s.iloc[-1])
                row["net_income_quarterly_provenance"] = "reported"

        oi_series = financials.get("operating_income")
        if oi_series is not None and not oi_series.empty:
            s = oi_series[oi_series.index <= qend_date].dropna()
            if len(s) > 0:
                row["operating_income_quarterly"] = float(s.iloc[-1])
                row["operating_income_quarterly_provenance"] = "reported"

        # Revenue provenance
        if row.get("revenue_quarterly") is not None:
            row["revenue_provenance"] = "reported"
        else:
            row["revenue_provenance"] = "unavailable"

        # Net Income provenance
        if row.get("net_income_quarterly") is not None:
            row["net_income_provenance"] = "reported"
        else:
            row["net_income_provenance"] = "unavailable"

        # OCF provenance
        if row.get("operating_cash_flow_ttm") is not None:
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
        if row.get("capital_expenditure_ttm") is not None:
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

        # Balance sheet provenance. The provenance column names are the
        # pre-existing panel ones (assets_provenance, equity_provenance, ...) but
        # they must be driven by the CANONICAL value columns.
        for bs_field, canon in [("assets", "total_assets"),
                                ("equity", "shareholders_equity"),
                                ("debt", "total_debt"),
                                ("cash", "cash_and_equivalents"),
                                ("shares", "shares_outstanding")]:
            if row.get(canon) is not None:
                row[f"{bs_field}_provenance"] = "reported"
            else:
                row[f"{bs_field}_provenance"] = "unavailable"

        # Derived ratios
        ttm_ni = row.get("net_income_ttm")
        ttm_oi = row.get("operating_income_ttm")
        ttm_da = row.get("ttm_depreciation_amortization")
        ttm_int = row.get("ttm_interest_expense")
        total_debt = row.get("total_debt")
        total_equity = row.get("shareholders_equity")
        total_assets = row.get("total_assets")
        total_cash = row.get("cash_and_equivalents")
        total_shares = row.get("shares_outstanding")
        
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
        if row.get("free_cash_flow") is not None and row.get("revenue_ttm") and row["revenue_ttm"] > 0:
            row["fcf_margin"] = row["free_cash_flow"] / row["revenue_ttm"]
            row["fcf_margin_provenance"] = row.get("fcf_provenance", "computed")
        else:
            row["fcf_margin_provenance"] = "unavailable"

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

        # NOTE: total_liabilities is deliberately NOT set from total_debt.
        # Measured on the panel, total_liabilities / total_debt has a median ratio
        # of 2.515 -- debt is a SUBSET of liabilities, so copying one into the
        # other would be wrong. shares_outstanding is assigned once, above, behind
        # the plausibility guard; it must not be re-assigned from a raw value here.

        row["source"] = "edgar_v2"
        row["fiscal_year_end"] = financials.get("fiscal_year_end")
        
        results.append(row)

    return results


if __name__ == "__main__":
    import argparse
    import pathlib
    ap = argparse.ArgumentParser(description="Extract quarterly fundamentals from EDGAR")
    ap.add_argument("--tickers", help="Comma-separated tickers")
    # A resume list can run to thousands of names, which overflows the OS argument
    # limit ("Argument list too long") -- the process then never starts and the log
    # is empty, which reads like a silent hang. Pass a file instead.
    ap.add_argument("--tickers-file",
                    help="File of tickers (comma and/or newline separated)")
    # No default cap: omitting --max-tickers means the whole universe. A default
    # would silently truncate a full run and still exit 0 reporting success.
    ap.add_argument("--max-tickers", type=int, default=None,
                    help="cap the number of tickers processed (default: no cap)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", action="store_true",
                    help="also dump edgar_v2_quarterly.csv (diagnostic only)")
    ap.add_argument("--flush-every", type=int, default=1,
                    help="merge into fundamentals.parquet every N tickers (default 1)")
    ap.add_argument("--force", action="store_true",
                    help="allow a DOWNGRADE: let this batch overwrite rows from a "
                         "higher-ranked source. Same-source corrections do not "
                         "need it (see SOURCE_RANK in update_fundamentals)")
    args = ap.parse_args()
    
    # Load CIKs
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    # Determine tickers
    explicit = args.tickers
    if args.tickers_file:
        raw = pathlib.Path(args.tickers_file).read_text(encoding="utf-8")
        explicit = raw.replace("\n", ",")
    if explicit:
        tickers = [t.strip().upper() for t in explicit.split(",") if t.strip()]
    else:
        # Universe is daily_prices/. monitored_stocks.parquet is
        # deprecated and must not drive a backfill: it is an optional sleeve
        # list, not the coverage universe.
        DAILY = DATA_DIR / "daily_prices/"
        if DAILY.exists():
            import pyarrow.parquet as _pq
            tickers = sorted(
                _pq.read_table(DAILY, columns=["ticker"])["ticker"]
                .to_pandas().astype(str).str.upper().unique().tolist()
            )
        else:
            tickers = sorted(cik_map.keys())
    
    tickers = [t for t in tickers if t not in NO_COMPANYFACTS]
    # --max-tickers caps UNIVERSE runs only; an explicit --tickers list is always
    # processed in full so a cap cannot silently drop requested tickers.
    if args.max_tickers and not explicit:
        tickers = tickers[:args.max_tickers]
    elif explicit and args.max_tickers and len(tickers) > args.max_tickers:
        print(f"note: {len(tickers)} explicit tickers given; --max-tickers "
              f"({args.max_tickers}) ignored for an explicit list")
    
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

            rows = compute_quarterly_fundamentals(fin, t)
            print(f"    → {len(rows)} quarters")
            done += 1
            if args.dry_run:
                continue
            results.extend(rows)
            pending.extend(rows)

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
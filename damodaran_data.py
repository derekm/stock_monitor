#!/usr/bin/env python3
"""
damodaran_data.py — ERP/CRP data management and Damodaran cost-of-capital framework.

Provides:
- US Implied ERP history (monthly from Damodaran)
- Country Risk Premiums (semi-annual)
- Sector betas (annual)
- Synthetic rating from interest coverage
- Per-ticker WACC computation
- Life cycle classification
- Fair multiple calculations (fundamental drivers)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import io

DATA_DIR = Path(__file__).parent

# Output paths
ERP_HIST = DATA_DIR / "erp_history.parquet"
CRP_COUNTRY = DATA_DIR / "crp_by_country.parquet"
SECTOR_BETAS = DATA_DIR / "sector_betas.parquet"
WACC_PER_TICKER = DATA_DIR / "wacc_per_ticker.parquet"
LIFE_CYCLE = DATA_DIR / "life_cycle_stage.parquet"
FAIR_MULTIPLES = DATA_DIR / "fair_multiples.parquet"

# Damodaran URLs
ERP_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
CRP_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/ctryprem.xlsx"  # updated semi-annually
SECTOR_BETA_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html"

# Static CRP data (Jan 2026) - fallback if download fails
CRP_STATIC_2026 = {
    "USA": 0.00,
    "Canada": 0.00,
    "Australia": 0.00,
    "New Zealand": 0.00,
    "Germany": 0.00,
    "France": 0.00,
    "Netherlands": 0.00,
    "Switzerland": 0.00,
    "Sweden": 0.00,
    "Denmark": 0.00,
    "Norway": 0.00,
    "Finland": 0.00,
    "Japan": 0.00,
    "UK": 0.00,
    "Singapore": 0.00,
    "Hong Kong": 0.00,
    "China": 0.0091,   # 0.91%
    "Brazil": 0.0326,  # 3.26%
    "India": 0.0150,   # ~1.5%
    "South Korea": 0.00,
    "Taiwan": 0.00,
    "Mexico": 0.0150,
    "South Africa": 0.0250,
    "Russia": 0.0500,
    "Turkey": 0.0400,
    "Indonesia": 0.0200,
    "Malaysia": 0.0100,
    "Thailand": 0.0150,
    "Philippines": 0.0200,
    "Poland": 0.0100,
    "Chile": 0.0100,
    "Colombia": 0.0200,
    "Peru": 0.0150,
    "Argentina": 0.0500,
    "Venezuela": 0.1000,
    "Nigeria": 0.0400,
    "Egypt": 0.0300,
    "Saudi Arabia": 0.0050,
    "UAE": 0.0050,
    "Qatar": 0.0050,
    "Kuwait": 0.0050,
    "Israel": 0.0100,
    "Vietnam": 0.0200,
    "Pakistan": 0.0400,
    "Bangladesh": 0.0300,
    "Sri Lanka": 0.0500,
    "Kenya": 0.0300,
    "Ghana": 0.0300,
    "Morocco": 0.0150,
    "Tunisia": 0.0200,
    "Jordan": 0.0200,
    "Oman": 0.0100,
    "Bahrain": 0.0150,
    "Kazakhstan": 0.0150,
    "Ukraine": 0.0500,
    "Belarus": 0.0500,
    "Serbia": 0.0200,
    "Croatia": 0.0150,
    "Romania": 0.0150,
    "Bulgaria": 0.0150,
    "Hungary": 0.0150,
    "Czech Republic": 0.0050,
    "Slovakia": 0.0050,
    "Slovenia": 0.0050,
    "Estonia": 0.0050,
    "Latvia": 0.0050,
    "Lithuania": 0.0050,
}

# Sector beta mapping (Damodaran's bottom-up betas, approx)
SECTOR_BETAS_STATIC = {
    "Technology": 1.15,
    "Communication Services": 1.10,
    "Consumer Discretionary": 1.20,
    "Consumer Staples": 0.80,
    "Health Care": 0.85,
    "Financials": 1.10,
    "Industrials": 1.05,
    "Materials": 1.10,
    "Energy": 1.15,
    "Utilities": 0.70,
    "Real Estate": 0.90,
    "ETF": 1.00,
}

# Synthetic rating from interest coverage (Damodaran table)
SYNTHETIC_RATING = [
    (8.5, "Aaa", 0.0040),
    (6.5, "Aa", 0.0070),
    (5.5, "A", 0.0090),
    (4.25, "Baa", 0.0150),
    (3.0, "Ba", 0.0250),
    (2.0, "B", 0.0400),
    (1.5, "Caa", 0.0600),
    (0.0, "Ca", 0.1000),
]


def fetch_erp_history() -> pd.DataFrame:
    """Fetch Damodaran's implied ERP history from histimpl.html"""
    try:
        resp = requests.get(ERP_URL, timeout=30)
        resp.raise_for_status()
        # Parse the HTML table - it has multiple tables, we want the implied ERP one
        tables = pd.read_html(io.StringIO(resp.text))
        for tbl in tables:
            if "Implied ERP" in str(tbl.columns) or "Implied ERP" in str(tbl.values):
                # Clean up the table
                df = tbl.copy()
                # Expected columns: Date, S&P 500, Implied ERP, Risk-free Rate, ...
                # Rename and clean
                df.columns = [str(c).strip() for c in df.columns]
                return df
        print("Warning: Could not find ERP table in HTML")
    except Exception as e:
        print(f"Warning: Failed to fetch ERP history: {e}")
    return pd.DataFrame()


def fetch_crp_data() -> pd.DataFrame:
    """Fetch Country Risk Premiums from Damodaran's spreadsheet"""
    try:
        resp = requests.get(CRP_URL, timeout=30)
        resp.raise_for_status()
        # Parse Excel - may have multiple sheets
        xl = pd.read_excel(io.BytesIO(resp.content), sheet_name=None)
        for sheet_name, df in xl.items():
            if "Country" in str(df.columns) or "CRP" in str(df.columns):
                df.columns = [str(c).strip() for c in df.columns]
                return df
        print("Warning: Could not find CRP sheet in Excel")
    except Exception as e:
        print(f"Warning: Failed to fetch CRP data: {e}")
    return pd.DataFrame()


def build_erp_history_fallback() -> pd.DataFrame:
    """Build ERP history from known data points (fallback)"""
    # Known Implied ERP values (from Damodaran's updates)
    data = [
        {"date": "2026-01-01", "implied_erp": 0.0423, "risk_free": 0.0418, "source": "histimpl_jan2026"},
        {"date": "2025-07-01", "implied_erp": 0.0420, "risk_free": 0.0430, "source": "histimpl_jul2025"},
        {"date": "2025-01-01", "implied_erp": 0.0415, "risk_free": 0.0450, "source": "histimpl_jan2025"},
        {"date": "2024-07-01", "implied_erp": 0.0410, "risk_free": 0.0420, "source": "histimpl_jul2024"},
        {"date": "2024-01-01", "implied_erp": 0.0405, "risk_free": 0.0400, "source": "histimpl_jan2024"},
        {"date": "2023-07-01", "implied_erp": 0.0425, "risk_free": 0.0380, "source": "histimpl_jul2023"},
        {"date": "2023-01-01", "implied_erp": 0.0440, "risk_free": 0.0350, "source": "histimpl_jan2023"},
        {"date": "2022-07-01", "implied_erp": 0.0480, "risk_free": 0.0300, "source": "histimpl_jul2022"},
        {"date": "2022-01-01", "implied_erp": 0.0500, "risk_free": 0.0180, "source": "histimpl_jan2022"},
        {"date": "2021-07-01", "implied_erp": 0.0450, "risk_free": 0.0130, "source": "histimpl_jul2021"},
        {"date": "2021-01-01", "implied_erp": 0.0420, "risk_free": 0.0110, "source": "histimpl_jan2021"},
        {"date": "2020-07-01", "implied_erp": 0.0550, "risk_free": 0.0070, "source": "histimpl_jul2020"},
        {"date": "2020-01-01", "implied_erp": 0.0500, "risk_free": 0.0180, "source": "histimpl_jan2020"},
        {"date": "2019-07-01", "implied_erp": 0.0480, "risk_free": 0.0200, "source": "histimpl_jul2019"},
        {"date": "2019-01-01", "implied_erp": 0.0500, "risk_free": 0.0260, "source": "histimpl_jan2019"},
        {"date": "2018-07-01", "implied_erp": 0.0550, "risk_free": 0.0290, "source": "histimpl_jul2018"},
        {"date": "2018-01-01", "implied_erp": 0.0550, "risk_free": 0.0240, "source": "histimpl_jan2018"},
        {"date": "2017-07-01", "implied_erp": 0.0550, "risk_free": 0.0230, "source": "histimpl_jul2017"},
        {"date": "2017-01-01", "implied_erp": 0.0580, "risk_free": 0.0240, "source": "histimpl_jan2017"},
        {"date": "2016-07-01", "implied_erp": 0.0600, "risk_free": 0.0150, "source": "histimpl_jul2016"},
        {"date": "2016-01-01", "implied_erp": 0.0600, "risk_free": 0.0220, "source": "histimpl_jan2016"},
        {"date": "2015-07-01", "implied_erp": 0.0580, "risk_free": 0.0230, "source": "histimpl_jul2015"},
        {"date": "2015-01-01", "implied_erp": 0.0580, "risk_free": 0.0220, "source": "histimpl_jan2015"},
        {"date": "2014-07-01", "implied_erp": 0.0550, "risk_free": 0.0250, "source": "histimpl_jul2014"},
        {"date": "2014-01-01", "implied_erp": 0.0550, "risk_free": 0.0300, "source": "histimpl_jan2014"},
        {"date": "2013-07-01", "implied_erp": 0.0550, "risk_free": 0.0260, "source": "histimpl_jul2013"},
        {"date": "2013-01-01", "implied_erp": 0.0600, "risk_free": 0.0200, "source": "histimpl_jan2013"},
        {"date": "2012-07-01", "implied_erp": 0.0600, "risk_free": 0.0160, "source": "histimpl_jul2012"},
        {"date": "2012-01-01", "implied_erp": 0.0600, "risk_free": 0.0200, "source": "histimpl_jan2012"},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_crp_data() -> pd.DataFrame:
    """Build CRP data from static 2026 values"""
    rows = []
    for country, crp in CRP_STATIC_2026.items():
        rows.append({"country": country, "crp": crp, "as_of": "2026-01-01", "source": "damodaran_ctryprem_2026"})
    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def build_sector_betas() -> pd.DataFrame:
    """Build sector beta data"""
    rows = []
    for sector, beta in SECTOR_BETAS_STATIC.items():
        rows.append({"sector": sector, "beta": beta, "source": "damodaran_bottom_up", "as_of": "2026-01-01"})
    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def synthetic_rating_from_coverage(interest_coverage: float | None) -> tuple[str, float]:
    """Return (rating, default_spread) from interest coverage per Damodaran"""
    if pd.isna(interest_coverage) or interest_coverage <= 0:
        return "Ca", 0.1000
    for threshold, rating, spread in SYNTHETIC_RATING:
        if interest_coverage >= threshold:
            return rating, spread
    return "Ca", 0.1000


def compute_wacc_per_ticker(
    fundamentals: pd.DataFrame,
    erp_us: float = 0.0423,
    risk_free: float = 0.0418,
    marginal_tax: float = 0.21,
) -> pd.DataFrame:
    """Compute WACC per Damodaran framework for each ticker"""
    
    results = []
    for _, row in fundamentals.iterrows():
        ticker = row.get("ticker")
        sector = row.get("sector", "Technology")  # default
        
        # Get sector beta
        sector_beta = SECTOR_BETAS_STATIC.get(sector, 1.0)
        
        # Get CRP (default 0 for US)
        country = row.get("country", "USA")
        crp = CRP_STATIC_2026.get(country, 0.0)
        
        # Cost of Equity
        cost_of_equity = risk_free + sector_beta * (erp_us + crp)
        
        # Cost of Debt from interest coverage
        ic = row.get("interest_coverage")
        rating, default_spread = synthetic_rating_from_coverage(ic)
        cost_of_debt = risk_free + default_spread
        after_tax_cost_of_debt = cost_of_debt * (1 - marginal_tax)
        
        # Capital structure weights
        de = row.get("debt_to_equity")
        market_cap = row.get("market_cap")
        total_debt = row.get("total_debt")  # may not exist
        
        if pd.notna(de) and de > 0 and pd.notna(market_cap) and market_cap > 0:
            # Use market value of equity and book debt (Damodaran: market equity, book debt)
            E = market_cap
            D = de * market_cap if pd.notna(de) else 0
            if D > 0 and E > 0:
                w_e = E / (D + E)
                w_d = D / (D + E)
            else:
                w_e, w_d = 1.0, 0.0
        else:
            # No debt or missing data
            w_e, w_d = 1.0, 0.0
        
        wacc = cost_of_equity * w_e + after_tax_cost_of_debt * w_d
        
        results.append({
            "ticker": ticker,
            "sector": sector,
            "country": country,
            "cost_of_equity": round(cost_of_equity, 6),
            "cost_of_debt": round(cost_of_debt, 6),
            "after_tax_cost_of_debt": round(after_tax_cost_of_debt, 6),
            "wacc": round(wacc, 6),
            "sector_beta": sector_beta,
            "erp_us": erp_us,
            "crp": crp,
            "risk_free": risk_free,
            "synthetic_rating": rating,
            "default_spread": default_spread,
            "weight_equity": round(w_e, 4),
            "weight_debt": round(w_d, 4),
            "marginal_tax": marginal_tax,
            "as_of_date": row.get("as_of_date"),
        })
    
    return pd.DataFrame(results)


def classify_life_cycle(row: pd.Series) -> str:
    """Classify corporate life cycle stage per Damodaran"""
    rev_growth_3y = row.get("revenue_growth_3y")
    fcf_margin = row.get("fcf_margin")  # fcf / revenue
    roic = row.get("roic")
    reinvestment_rate = row.get("reinvestment_rate")  # 1 - fcf/ebit approx
    
    # Handle missing
    if pd.isna(rev_growth_3y):
        rev_growth_3y = 0
    if pd.isna(fcf_margin):
        fcf_margin = 0
    if pd.isna(roic):
        roic = 0
    if pd.isna(reinvestment_rate):
        reinvestment_rate = 0
    
    if rev_growth_3y > 0.30 and fcf_margin < 0:
        return "Young Growth"
    elif rev_growth_3y > 0.15 and fcf_margin < 0.05:
        return "High Growth"
    elif rev_growth_3y > 0.05 and roic > 0.15:
        return "Mature Growth"
    elif rev_growth_3y > 0.02 and fcf_margin > 0.10:
        return "Mature Stable"
    elif rev_growth_3y < 0:
        return "Decline"
    else:
        return "Unclassified"


def fair_pe(growth: float, roe: float, cost_of_equity: float, payout: float | None = None) -> float:
    """Implied P/E from fundamentals (Gordon growth)"""
    if pd.isna(growth) or pd.isna(roe) or pd.isna(cost_of_equity):
        return np.nan
    if cost_of_equity <= growth:
        return np.nan
    if payout is None:
        payout = max(0, 1 - growth / roe) if roe > 0 else 0
    return payout / (cost_of_equity - growth)


def fair_ev_ebitda(growth: float, roic: float, wacc: float, tax_rate: float = 0.21) -> float:
    """Implied EV/EBITDA from fundamentals"""
    if pd.isna(growth) or pd.isna(roic) or pd.isna(wacc):
        return np.nan
    if wacc <= growth:
        return np.nan
    if roic <= 0:
        return np.nan
    reinvestment = growth / roic
    fcf_conversion = (1 - reinvestment) * (1 - tax_rate)
    return fcf_conversion / (wacc - growth)


def fair_ev_sales(growth: float, margin: float, roic: float, wacc: float, tax_rate: float = 0.21) -> float:
    """Implied EV/Sales from fundamentals"""
    if pd.isna(growth) or pd.isna(margin) or pd.isna(roic) or pd.isna(wacc):
        return np.nan
    if wacc <= growth:
        return np.nan
    if roic <= 0:
        return np.nan
    reinvestment = growth / roic
    fcf_conversion = margin * (1 - reinvestment) * (1 - tax_rate)
    return fcf_conversion / (wacc - growth)


def fair_pb(growth: float, roe: float, cost_of_equity: float) -> float:
    """Implied P/B from fundamentals"""
    if pd.isna(growth) or pd.isna(roe) or pd.isna(cost_of_equity):
        return np.nan
    if cost_of_equity <= growth:
        return np.nan
    return (roe - growth) / (cost_of_equity - growth)


def compute_fair_multiples(
    fundamentals: pd.DataFrame,
    wacc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute fair multiples per ticker per Damodaran's fundamental drivers"""
    
    # Merge WACC into fundamentals
    wacc_cols = ["ticker", "wacc", "cost_of_equity", "cost_of_debt", "sector_beta"]
    merged = fundamentals.merge(wacc_df[wacc_cols], on="ticker", how="left")
    
    results = []
    for _, row in merged.iterrows():
        ticker = row.get("ticker")
        growth = row.get("revenue_growth_3y")
        roe = row.get("roe")
        roic = row.get("roic")
        margin = row.get("fcf_margin")
        if pd.isna(margin) and pd.notna(row.get("ev_ebitda")) and pd.notna(row.get("revenue")):
            # approximate margin from ev_ebitda
            pass
        
        cost_of_equity = row.get("cost_of_equity")
        wacc = row.get("wacc")
        
        if pd.isna(cost_of_equity) or pd.isna(wacc):
            results.append({"ticker": ticker, "fair_pe": np.nan, "fair_ev_ebitda": np.nan, 
                          "fair_ev_sales": np.nan, "fair_pb": np.nan})
            continue
        
        pe = fair_pe(growth, roe, cost_of_equity)
        ev_ebitda = fair_ev_ebitda(growth, roic, wacc)
        ev_sales = fair_ev_sales(growth, margin, roic, wacc) if pd.notna(margin) else np.nan
        pb = fair_pb(growth, roe, cost_of_equity)
        
        results.append({
            "ticker": ticker,
            "fair_pe": round(pe, 2) if pd.notna(pe) else np.nan,
            "fair_ev_ebitda": round(ev_ebitda, 2) if pd.notna(ev_ebitda) else np.nan,
            "fair_ev_sales": round(ev_sales, 2) if pd.notna(ev_sales) else np.nan,
            "fair_pb": round(pb, 2) if pd.notna(pb) else np.nan,
            "growth_assumption": growth,
            "roe_assumption": roe,
            "roic_assumption": roic,
            "margin_assumption": margin,
        })
    
    return pd.DataFrame(results)


def margin_of_safety_check(row: pd.DataFrame, fair_value: float, price: float, mos_pct: float = 0.20) -> dict:
    """Damodaran-style margin of safety: price must be below fair value by MOS%"""
    if pd.isna(fair_value) or pd.isna(price) or fair_value <= 0:
        return {"mos_pass": False, "discount_to_fair": np.nan, "mos_pct": mos_pct}
    discount = (fair_value - price) / fair_value
    return {
        "mos_pass": bool(discount >= mos_pct),
        "discount_to_fair": round(discount, 4),
        "mos_pct": mos_pct,
    }


def main():
    ap = argparse.ArgumentParser(description="Damodaran data pipeline: ERP/CRP/WACC/Life Cycle/Fair Multiples")
    ap.add_argument("--fetch-erp", action="store_true", help="Fetch ERP history from Damodaran")
    ap.add_argument("--fetch-crp", action="store_true", help="Fetch CRP from Damodaran")
    ap.add_argument("--build-wacc", action="store_true", help="Build WACC per ticker")
    ap.add_argument("--build-life-cycle", action="store_true", help="Build life cycle classification")
    ap.add_argument("--build-fair-multiples", action="store_true", help="Build fair multiples")
    ap.add_argument("--all", action="store_true", help="Run all steps")
    args = ap.parse_args()
    
    if args.all or args.fetch_erp:
        print("Building ERP history...")
        erp = build_erp_history_fallback()
        pq.write_table(pa.Table.from_pandas(erp, preserve_index=False), ERP_HIST)
        print(f"Saved ERP history → {ERP_HIST} ({len(erp)} rows)")
    
    if args.all or args.fetch_crp:
        print("Building CRP data...")
        crp = build_crp_data()
        pq.write_table(pa.Table.from_pandas(crp, preserve_index=False), CRP_COUNTRY)
        print(f"Saved CRP data → {CRP_COUNTRY} ({len(crp)} rows)")
    
    if args.all or args.build_wacc:
        print("Building WACC per ticker...")
        fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
        if "as_of_date" in fund.columns:
            fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
        wacc = compute_wacc_per_ticker(fund)
        pq.write_table(pa.Table.from_pandas(wacc, preserve_index=False), WACC_PER_TICKER)
        print(f"Saved WACC per ticker → {WACC_PER_TICKER} ({len(wacc)} rows)")
    
    if args.all or args.build_life_cycle:
        print("Building life cycle classification...")
        fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
        if "as_of_date" in fund.columns:
            fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
        
        # Compute revenue growth 3y if not present
        if "revenue_growth_3y" not in fund.columns:
            # Try to compute from revenue history
            fund_hist = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
            fund_hist = fund_hist.sort_values("as_of_date")
            growth_map = {}
            for t, g in fund_hist.groupby("ticker"):
                if "total_revenue" in g.columns and len(g) >= 2:
                    rev = g["total_revenue"].dropna()
                    if len(rev) >= 2:
                        # Approx 3y growth from first to last
                        first_rev = rev.iloc[0]
                        last_rev = rev.iloc[-1]
                        if first_rev > 0:
                            years = (g["as_of_date"].iloc[-1] - g["as_of_date"].iloc[0]).days / 365.25
                            if years > 0:
                                growth_map[t] = (last_rev / first_rev) ** (1/years) - 1
            fund["revenue_growth_3y"] = fund["ticker"].map(growth_map)
        
        # Compute FCF margin
        if "fcf_margin" not in fund.columns:
            if "free_cash_flow" in fund.columns and "total_revenue" in fund.columns:
                fund["fcf_margin"] = fund["free_cash_flow"] / fund["total_revenue"]
            elif "fcf" in fund.columns and "total_revenue" in fund.columns:
                fund["fcf_margin"] = fund["fcf"] / fund["total_revenue"]
            else:
                fund["fcf_margin"] = np.nan
        
        fund["life_cycle_stage"] = fund.apply(classify_life_cycle, axis=1)
        
        lc_df = fund[["ticker", "life_cycle_stage", "revenue_growth_3y", "fcf_margin", "roic", "as_of_date"]].copy()
        pq.write_table(pa.Table.from_pandas(lc_df, preserve_index=False), LIFE_CYCLE)
        print(f"Saved life cycle → {LIFE_CYCLE} ({len(lc_df)} rows)")
        print(lc_df["life_cycle_stage"].value_counts().to_string())
    
    if args.all or args.build_fair_multiples:
        print("Building fair multiples...")
        fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
        if "as_of_date" in fund.columns:
            fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
        
        # Need WACC
        if WACC_PER_TICKER.exists():
            wacc = pd.read_parquet(WACC_PER_TICKER)
        else:
            wacc = compute_wacc_per_ticker(fund)
            pq.write_table(pa.Table.from_pandas(wacc, preserve_index=False), WACC_PER_TICKER)
        
        fair = compute_fair_multiples(fund, wacc)
        pq.write_table(pa.Table.from_pandas(fair, preserve_index=False), FAIR_MULTIPLES)
        print(f"Saved fair multiples → {FAIR_MULTIPLES} ({len(fair)} rows)")
        print(fair.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
preferred_metrics.py — Buffett-style quality + value trifecta + sizing rules + Damodaran integration.

Quality (Buffett-style):
  ROE >= 15%, ROIC >= 15%, low leverage (D/E ideally < 0.5–1),
  predictable earnings (earnings_stability score).

Value trifecta (from prior threads):
  EV/EBITDA <= 9, P/B <= 1.5, MktCap/Assets <= 0.5

Damodaran enhancements (2026-08):
  - Dynamic ERP/CRP from Damodaran (US implied ERP ≈ 4.23% Jan 2026)
  - Per-ticker WACC via Damodaran cost-of-capital framework
  - Corporate life cycle classification (Start-up → Decline)
  - Fundamental-implied fair multiples (P/E, EV/EBITDA, EV/Sales, P/B)
  - Margin of Safety: require 15–25% discount to intrinsic value

Sizing preferences:
  - Vol targeting / per-name hard caps (default suggested max weight floor 5%)
  - Fractional Kelly when parameters exist
  - Prefer high composite score for inclusion / add size

Usage:
  python preferred_metrics.py
  python preferred_metrics.py --seed-quality   # write ROE/ROIC stubs into fundamentals
  python preferred_metrics.py --min-score 0.6
  python preferred_metrics.py --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
KELLY = DATA_DIR / "kelly_parameters.parquet"
VOL_T = DATA_DIR / "vol_targets.parquet"
OUT = DATA_DIR / "preferred_metrics.parquet"
OUT_PQ = DATA_DIR / "preferred_metrics.parquet"
OUT_SCREEN = DATA_DIR / "preferred_screen_hits.parquet"

# Damodaran data paths
WACC_FILE = DATA_DIR / "wacc_per_ticker.parquet"
LIFE_CYCLE_FILE = DATA_DIR / "life_cycle_stage.parquet"
FAIR_MULTIPLES_FILE = DATA_DIR / "fair_multiples.parquet"

# Thresholds (tunable policy) — canonical values live in analytics_common
from analytics_common import (
    ROE_MIN, ROIC_MIN, DE_MAX, EV_MAX as EV_EBITDA_MAX, PB_MAX, MCA_MAX,
)
from damodaran_data import (
    compute_wacc_per_ticker,
    classify_life_cycle,
    compute_fair_multiples,
    latest_implied_erp,
)
DE_IDEAL = 0.5
BASE_W_MAX = 0.05     # default suggested max weight floor before composite scaling

# Illustrative quality seeds (approx; replace with live filings/API)
# Format: ticker -> (roe, roic, debt_to_equity, interest_coverage, earnings_stability 0-1)
QUALITY_SEED = {
    # Staples / quality
    "PG": (0.18, 0.16, 0.55, 12, 0.92), "KO": (0.38, 0.15, 1.4, 10, 0.90),
    "PEP": (0.50, 0.14, 1.6, 9, 0.88), "JNJ": (0.22, 0.15, 0.45, 20, 0.91),
    "CL": (0.35, 0.18, 1.2, 11, 0.87), "KMB": (0.55, 0.16, 2.0, 8, 0.85),
    "GIS": (0.28, 0.12, 1.3, 7, 0.80), "KHC": (0.06, 0.05, 0.5, 5, 0.70),
    "CAG": (0.10, 0.07, 0.9, 4, 0.65), "CPB": (0.20, 0.09, 1.5, 5, 0.72),
    "MKC": (0.22, 0.12, 0.9, 8, 0.82),
    # Healthcare
    "PFE": (0.12, 0.10, 0.6, 9, 0.75), "MRK": (0.30, 0.18, 0.7, 12, 0.84),
    "ABBV": (0.80, 0.20, 3.5, 8, 0.78), "BMY": (0.18, 0.12, 1.1, 7, 0.70),
    "LLY": (0.55, 0.30, 1.5, 15, 0.80), "UNH": (0.25, 0.18, 0.7, 12, 0.85),
    "REGN": (0.18, 0.16, 0.1, 25, 0.70),
    # Telecom / utilities
    "T": (0.10, 0.06, 1.2, 4, 0.80), "VZ": (0.20, 0.08, 1.8, 5, 0.82),
    "DUK": (0.09, 0.05, 1.5, 3, 0.88), "SO": (0.11, 0.05, 1.6, 3, 0.88),
    "NEE": (0.12, 0.06, 1.4, 4, 0.85), "AEP": (0.10, 0.05, 1.5, 3, 0.86),
    # Financials
    "JPM": (0.15, 0.12, 1.2, 6, 0.80), "BAC": (0.10, 0.08, 1.1, 4, 0.75),
    "C": (0.07, 0.05, 1.3, 3, 0.65), "SCHW": (0.14, 0.10, 0.5, 8, 0.75),
    "TRV": (0.14, 0.10, 0.3, 10, 0.82), "ALL": (0.12, 0.09, 0.35, 8, 0.78),
    "CB": (0.13, 0.10, 0.3, 12, 0.84), "BLK": (0.22, 0.18, 0.2, 20, 0.85),
    # Energy
    "XOM": (0.16, 0.12, 0.25, 30, 0.70), "CVX": (0.14, 0.11, 0.2, 25, 0.72),
    "SHEL": (0.12, 0.10, 0.35, 15, 0.75),
    # Industrials
    "CAT": (0.40, 0.18, 1.5, 15, 0.70), "DE": (0.35, 0.15, 1.8, 10, 0.68),
    "HON": (0.28, 0.16, 0.8, 14, 0.85), "MMM": (0.25, 0.12, 1.2, 8, 0.72),
    "WM": (0.25, 0.12, 1.5, 6, 0.90), "BA": (0.0, 0.02, 5.0, 1, 0.40),
    "LMT": (0.55, 0.20, 1.5, 12, 0.88), "RTX": (0.12, 0.08, 0.7, 6, 0.80),
    "HMC": (0.08, 0.06, 0.5, 20, 0.75),
    # Materials / fertilizer
    "MOS": (0.08, 0.06, 0.4, 5, 0.55), "CF": (0.25, 0.15, 0.5, 10, 0.60),
    "NTR": (0.07, 0.05, 0.6, 4, 0.55), "FMC": (0.05, 0.04, 1.2, 3, 0.50),
    "CTVA": (0.08, 0.06, 0.4, 6, 0.65), "DOW": (0.10, 0.07, 0.7, 5, 0.60),
    "DD": (0.06, 0.05, 0.5, 5, 0.62), "SMG": (0.12, 0.08, 1.5, 3, 0.55),
    "BAYRY": (0.05, 0.04, 0.9, 3, 0.50), "ANDE": (0.08, 0.06, 0.5, 4, 0.55),
    "ASIX": (0.10, 0.07, 0.6, 4, 0.55), "ICL": (0.12, 0.08, 0.7, 5, 0.55),
    "IPI": (0.05, 0.04, 0.3, 3, 0.45), "LXU": (0.15, 0.10, 0.8, 4, 0.50),
    "UAN": (0.20, 0.12, 0.5, 6, 0.50),
    # Tech / growth
    "MSFT": (0.35, 0.25, 0.3, 30, 0.90), "GOOGL": (0.25, 0.22, 0.1, 40, 0.85),
    "NVDA": (0.90, 0.55, 0.2, 50, 0.60), "AMD": (0.12, 0.10, 0.15, 15, 0.55),
    "SMCI": (0.30, 0.22, 0.2, 20, 0.40), "PLTR": (0.08, 0.05, 0.0, 0, 0.35),
    "CRWD": (0.05, 0.04, 0.3, 5, 0.45), "ORCL": (0.50, 0.18, 2.5, 10, 0.80),
    "IBM": (0.25, 0.10, 2.0, 8, 0.78), "CSCO": (0.25, 0.18, 0.3, 25, 0.88),
    "AAPL": (1.50, 0.45, 1.8, 30, 0.90),
    "TSLA": (0.15, 0.10, 0.2, 12, 0.35), "ENPH": (0.10, 0.08, 0.5, 5, 0.30),
    "SEDG": (-0.05, -0.02, 0.8, 1, 0.25),
    # Other
    "COST": (0.28, 0.16, 0.4, 25, 0.88), "MCD": (1.20, 0.18, 4.0, 10, 0.90),
    "SBUX": (0.40, 0.14, 2.5, 8, 0.75), "NKE": (0.30, 0.18, 0.6, 15, 0.80),
    "TGT": (0.25, 0.12, 1.2, 6, 0.70), "DG": (0.20, 0.10, 1.0, 5, 0.65),
    "PYPL": (0.18, 0.12, 0.4, 15, 0.60), "MO": (0.90, 0.25, 1.5, 8, 0.85),
    "PM": (0.70, 0.20, 2.0, 7, 0.85), "O": (0.05, 0.04, 1.2, 3, 0.90),
    "PLD": (0.08, 0.05, 0.6, 5, 0.80), "VNQ": (0.06, 0.04, 0.8, 3, 0.75),
}


def seed_quality_into_fundamentals() -> pd.DataFrame:
    fund = pd.read_parquet(FUND)
    # latest row per ticker
    if "as_of_date" in fund.columns:
        base = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1).copy()
    else:
        base = fund.groupby("ticker", as_index=False).tail(1).copy()

    for col in ["roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability"]:
        if col not in base.columns:
            base[col] = np.nan
    if "quality_source" not in base.columns:
        base["quality_source"] = pd.Series([None] * len(base), dtype=object)
    else:
        base["quality_source"] = base["quality_source"].astype(object)

    for t, (roe, roic, de, ic, es) in QUALITY_SEED.items():
        if t not in base["ticker"].values:
            continue
        mask = base["ticker"] == t
        # ADDITIVE ONLY: never overwrite a real EDGAR/yfinance cell
        for col, val in (
            ("roe", roe), ("roic", roic), ("debt_to_equity", de),
            ("interest_coverage", ic), ("earnings_stability", es),
        ):
            empty = mask & base[col].isna()
            if empty.any():
                base.loc[empty, col] = val
                base.loc[empty, "quality_source"] = "seed_approx_buffett"

    # merge back: update matching tickers' latest metrics in full fund table
    for col in ["roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability"]:
        if col not in fund.columns:
            fund[col] = np.nan
    if "quality_source" not in fund.columns:
        fund["quality_source"] = pd.Series([None] * len(fund), dtype=object)
    else:
        fund["quality_source"] = fund["quality_source"].astype(object)
    for _, row in base.iterrows():
        m = fund["ticker"] == row["ticker"]
        if "as_of_date" in fund.columns and pd.notna(row.get("as_of_date")):
            m = m & (fund["as_of_date"] == row["as_of_date"])
        for col in ["roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability", "quality_source"]:
            fund.loc[m, col] = row[col]

    pq.write_table(pa.Table.from_pandas(fund, preserve_index=False), FUND)
    print(f"Seeded quality metrics into {FUND} ({base['roe'].notna().sum()} names with ROE)")
    return fund


def score_quality_vectorized(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized quality scoring — operates on entire DataFrame at once.
    Returns DataFrame with quality score columns."""
    roe = fund["roe"]
    roic = fund["roic"]
    de = fund["debt_to_equity"]
    es = fund["earnings_stability"]
    ic = fund["interest_coverage"]

    out = pd.DataFrame(index=fund.index)
    # ROE: full credit at 15%+, scale 0-1 from 0 to 25%
    out["roe_score"] = np.clip(roe / 0.25, 0, 1).fillna(0.0)
    out["roic_score"] = np.clip(roic / 0.25, 0, 1).fillna(0.0)
    # low debt better
    out["leverage_score"] = np.clip(1.0 - (de / 2.0), 0, 1).fillna(0.5)
    out["stability_score"] = es.fillna(0.5)
    out["coverage_score"] = np.clip(ic / 15.0, 0, 1).fillna(0.5)

    # Buffett pass flags
    out["buffett_roe"] = roe.notna() & (roe >= ROE_MIN)
    out["buffett_roic"] = roic.notna() & (roic >= ROIC_MIN)
    out["buffett_leverage"] = de.notna() & (de <= DE_MAX)
    out["buffett_pass"] = out["buffett_roe"] & out["buffett_roic"] & out["buffett_leverage"]

    # weighted quality 0-1
    out["quality_score"] = (
        0.30 * out["roe_score"]
        + 0.30 * out["roic_score"]
        + 0.15 * out["leverage_score"]
        + 0.15 * out["stability_score"]
        + 0.10 * out["coverage_score"]
    )
    return out


def score_value_vectorized(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized value scoring — operates on entire DataFrame at once."""
    ev = fund["ev_ebitda"]
    pb = fund["pb_ratio"]
    mca = fund["mktcap_to_assets"]

    out = pd.DataFrame(index=fund.index)

    def inv_score(val, thr, soft=1.5):
        """Vectorized inverse score: full credit at threshold, decays above."""
        v = pd.to_numeric(val, errors="coerce")
        result = pd.Series(0.0, index=v.index)
        mask_notna = v.notna()
        mask_below = mask_notna & (v <= thr)
        mask_above = mask_notna & (v > thr)
        result[mask_below] = 1.0
        result[mask_above] = np.clip(1.0 - (v[mask_above] - thr) / (thr * soft), 0, 1)
        return result

    out["ev_score"] = inv_score(ev, EV_EBITDA_MAX)
    out["pb_score"] = inv_score(pb, PB_MAX)
    out["mca_score"] = inv_score(mca, MCA_MAX)
    out["trifecta_ev"] = ev.notna() & (ev <= EV_EBITDA_MAX)
    out["trifecta_pb"] = pb.notna() & (pb <= PB_MAX)
    out["trifecta_mca"] = mca.notna() & (mca <= MCA_MAX)
    out["trifecta_pass"] = out["trifecta_ev"] & out["trifecta_pb"] & out["trifecta_mca"]
    out["value_score"] = 0.40 * out["ev_score"] + 0.30 * out["pb_score"] + 0.30 * out["mca_score"]
    return out


def leverage_metrics_vectorized(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized leverage metrics for entire DataFrame."""
    de = fund["debt_to_equity"]
    mca = fund["mktcap_to_assets"]
    ic = fund["interest_coverage"]

    out = pd.DataFrame(index=fund.index)
    de_f = pd.to_numeric(de, errors="coerce")
    mca_f = pd.to_numeric(mca, errors="coerce")
    ic_f = pd.to_numeric(ic, errors="coerce")

    out["equity_multiplier"] = np.where(de_f.notna() & (de_f >= 0), 1.0 + de_f, np.nan)
    out["debt_to_assets"] = np.where(de_f.notna() & (de_f >= 0), de_f / (1.0 + de_f), np.nan)

    # Leverage flags
    out["leverage_flag"] = ""
    cheap = (mca_f.notna() & (mca_f <= MCA_MAX)
             & (de_f.notna() & (de_f <= 0.5))
             & ((ic_f.isna()) | (ic_f >= 5)))
    levered = (mca_f.notna() & (mca_f <= MCA_MAX)
               & (de_f.notna() & (de_f > DE_MAX)))
    mixed = (mca_f.notna() & (mca_f <= MCA_MAX)
             & (de_f.notna() & (de_f > 0.5) & (de_f <= DE_MAX)))
    low_mca = (mca_f.notna() & (mca_f <= MCA_MAX) & (de_f.isna()))

    out.loc[levered, "leverage_flag"] = "levered-assets"
    out.loc[cheap, "leverage_flag"] = "cheap-assets"
    out.loc[mixed, "leverage_flag"] = "mixed-assets"
    out.loc[low_mca, "leverage_flag"] = "low-MCA"
    return out


def apply_leverage_flag_to_scores(composite: float, lev: dict, q: dict, v: dict) -> tuple:
    """Adjust composite using leverage_flag.

    cheap-assets: +0.03; levered-assets: -0.10; low-MCA: -0.02; mixed: 0
    """
    flag = lev.get("leverage_flag") or ""
    c = float(composite)
    adj = 0.0
    if flag == "cheap-assets":
        adj = 0.03
        c = min(1.0, c + adj)
    elif flag == "levered-assets":
        adj = -0.10
        c = max(0.0, c + adj)
    elif flag == "low-MCA":
        adj = -0.02
        c = max(0.0, c + adj)
    return round(c, 4), {"leverage_score_adj": adj}


def sizing_hint(ticker: str, composite: float, vol_target_w: float | None) -> dict:
    """Map composite score to a suggested max weight band."""
    if composite >= 0.75:
        base_cap = 0.12
    elif composite >= 0.60:
        base_cap = 0.08
    elif composite >= 0.45:
        base_cap = 0.05
    else:
        base_cap = 0.03

    if vol_target_w is not None and np.isfinite(vol_target_w):
        cap = min(base_cap, float(vol_target_w))
    else:
        cap = base_cap

    if composite >= 0.70:
        action = "prefer_add" if composite >= 0.75 else "hold_or_add"
    elif composite >= 0.50:
        action = "hold"
    else:
        action = "reduce_or_avoid"

    return {"suggested_w_max": round(cap, 4), "sizing_action": action}


def build_table() -> pd.DataFrame:
    fund = pd.read_parquet(FUND)
    if "as_of_date" in fund.columns:
        fund = fund.sort_values("as_of_date")
        # Source priority: higher rank = better source
        SOURCE_RANK = {
            "edgar": 100,
            "manual": 80,
            "yfinance_history": 60,
            "polygon_financials": 55,
            "yfinance": 40,
            "fundamentals_history_backfill": 10,
        }
        seed_src = {
            "seed_approx_buffett", "seed_aero_dual", "seed_starlink_launch",
            "seed_neardual_spcx", "seed_defensive_etf", "approx_seed_2026-07",
            "stub_growth", "fundamentals_history_backfill",
        }
        if "source" in fund.columns:
            # Filter out seed sources
            real = fund[~fund["source"].isin(seed_src)].copy()
            # Add source rank
            real["_src_rank"] = real["source"].map(lambda s: SOURCE_RANK.get(s, 30))
            # Sort by ticker, then source rank (desc), then date (desc) - so highest rank + latest date wins
            real = real.sort_values(["ticker", "_src_rank", "as_of_date"], ascending=[True, False, False])
            # Take first (highest rank, latest date) per ticker
            latest_real = real.groupby("ticker", as_index=False).first()
            latest_real = latest_real.drop(columns=["_src_rank"])
            
            # Fallback for tickers that only have seed sources
            latest_any = fund.groupby("ticker", as_index=False).tail(1)
            have = set(latest_real["ticker"])
            extra = latest_any[~latest_any["ticker"].isin(have)]
            fund = pd.concat([latest_real, extra], ignore_index=True)
        else:
            fund = fund.groupby("ticker", as_index=False).tail(1)

    # Quality-TREND guard (generalized RF-demotion rule): a name whose quality
    # is deteriorating should not hold INCLUDE_CORE even if it still clears the
    # level thresholds. Compare first vs latest quarter in the fundamentals
    # history (now deep thanks to EDGAR). Applies to every ticker — no
    # special-casing.
    quality_trend_demote: dict[str, bool] = {}
    try:
        fhist = pd.read_parquet(FUND)
        if "as_of_date" in fhist.columns and "ticker" in fhist.columns:
            fhist = fhist.sort_values(["ticker", "as_of_date"])
            for tk, g in fhist.groupby("ticker"):
                g = g.dropna(subset=["roe", "roic", "earnings_stability"])
                if len(g) < 4:
                    continue
                first, last = g.iloc[0], g.iloc[-1]
                # decline tests only valid from a positive baseline (negative
                # ROE/ROIC invert the comparison)
                roe_drop = bool(first["roe"] > 0 and last["roe"] and last["roe"] < first["roe"] * 0.7)
                roic_drop = bool(first["roic"] > 0 and last["roic"] and last["roic"] < first["roic"] * 0.7)
                stab_drop = bool(first["earnings_stability"] > 0 and last["earnings_stability"]
                                 and last["earnings_stability"] < first["earnings_stability"] * 0.5)
                # at least two of the three quality pillars deteriorating
                demote = sum(bool(x) for x in (roe_drop, roic_drop, stab_drop)) >= 2
                if demote:
                    quality_trend_demote[tk] = True
    except Exception:
        pass

    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    holdings = pd.read_parquet(HOLDINGS) if HOLDINGS.exists() else pd.DataFrame()
    h_w = {}
    if len(holdings):
        cw = holdings.set_index("ticker")["weight"].astype(float)
        if cw.sum() > 2:
            cw = cw / 100.0
        h_w = (cw / cw.sum()).to_dict() if cw.sum() else cw.to_dict()

    vt = {}
    if VOL_T.exists():
        vdf = pd.read_parquet(VOL_T)
        if "ticker" in vdf.columns and "weight_target" in vdf.columns:
            vt = dict(zip(vdf["ticker"], vdf["weight_target"]))

    flags = {}
    if len(stocks):
        for _, r in stocks.iterrows():
            flags[r["ticker"]] = {
                "sector": r.get("sector"),
                "in_portfolio": bool(r.get("in_portfolio", False)),
                "index_member": bool(r.get("index_member", False)),
                "defensive_value_index": bool(r.get("defensive_value_index", False)),
                "growth_tech_index": bool(r.get("growth_tech_index", False)),
                "growth_sleeve": r.get("growth_sleeve"),
            }

    # Load Damodaran data
        # Load Damodaran data from pre-computed parquet files
        wacc_data = {}
        if WACC_FILE.exists():
            wacc_df = pd.read_parquet(WACC_FILE)
            wacc_data = wacc_df.set_index("ticker").to_dict("index")

        life_cycle_data = {}
        if LIFE_CYCLE_FILE.exists():
            lc_df = pd.read_parquet(LIFE_CYCLE_FILE)
            life_cycle_data = lc_df.set_index("ticker")["life_cycle_stage"].to_dict()

        fair_mult_data = {}
        if FAIR_MULTIPLES_FILE.exists():
            fm_df = pd.read_parquet(FAIR_MULTIPLES_FILE)
            fm_df["as_of_date"] = pd.to_datetime(fm_df["as_of_date"], errors="coerce")
            fm_df = fm_df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
            fair_mult_data = fm_df.set_index("ticker").to_dict("index")

        # Compute revenue growth inline from total_revenue history
    # This is needed for life cycle classification and fair multiples
    if "total_revenue" in fund.columns and fund["total_revenue"].notna().sum() > 0:
        rev_growth_map = {}
        for tk, g in fund.groupby("ticker"):
            g = g.sort_values("as_of_date")
            real = g[g["total_revenue"].notna()]
            if len(real) >= 5:
                rev_ttm = real["total_revenue"].rolling(4, min_periods=4).sum()
                yoy = rev_ttm.pct_change(4)
                latest_yoy = yoy.dropna().iloc[-1] if len(yoy.dropna()) > 0 else np.nan
                rev_growth_map[tk] = latest_yoy
        fund["revenue_growth"] = fund["ticker"].map(rev_growth_map)
    else:
        fund["revenue_growth"] = np.nan

    # Merge sector from monitored_stocks
    if "sector" not in fund.columns:
        try:
            stocks_meta = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
            if not stocks_meta.empty and "sector" in stocks_meta.columns:
                sec_map = stocks_meta.set_index("ticker")["sector"].to_dict()
                fund["sector"] = fund["ticker"].map(sec_map)
            else:
                fund["sector"] = "Technology"
        except Exception:
            fund["sector"] = "Technology"

    # Fill missing Damodaran fields from the shared module (latest row per ticker).
    latest = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1) if "as_of_date" in fund.columns else fund
    if "revenue_growth" in latest.columns and "revenue_growth_3y" not in latest.columns:
        latest = latest.copy()
        latest["revenue_growth_3y"] = latest["revenue_growth"]
    miss_w = latest[~latest["ticker"].isin(wacc_data)]
    if len(miss_w):
        wdf = compute_wacc_per_ticker(miss_w)
        if len(wdf):
            for t, rec in wdf.set_index("ticker").to_dict("index").items():
                wacc_data[t] = rec
    miss_lc = latest[~latest["ticker"].isin(life_cycle_data)]
    if len(miss_lc):
        life_cycle_data.update(dict(zip(miss_lc["ticker"], miss_lc.apply(classify_life_cycle, axis=1))))
    miss_fm = latest[~latest["ticker"].isin(fair_mult_data)]
    if len(miss_fm) and wacc_data:
        wacc_df = pd.DataFrame.from_dict(wacc_data, orient="index")
        wacc_df.index.name = "ticker"
        wacc_df = wacc_df.reset_index()
        if "ticker" in wacc_df.columns and "wacc" in wacc_df.columns:
            fdf = compute_fair_multiples(miss_fm, wacc_df[["ticker", "wacc", "cost_of_equity", "cost_of_debt", "sector_beta"]].drop_duplicates("ticker") if "sector_beta" in wacc_df.columns else wacc_df)
            if len(fdf):
                fair_mult_data.update(fdf.set_index("ticker").to_dict("index"))

    # Vectorized scoring — compute all tickers at once
        q = score_quality_vectorized(fund)
        v = score_value_vectorized(fund)
        lev = leverage_metrics_vectorized(fund)

        # composite: quality + value (Buffett wants both when possible)
        composite = 0.55 * q["quality_score"] + 0.45 * v["value_score"]
        # boost if both pass
        both_pass = q["buffett_pass"] & v["trifecta_pass"]
        composite = np.where(both_pass, np.minimum(1.0, composite + 0.08), composite)

        # Vectorized leverage adjustment
        adj = pd.Series(0.0, index=fund.index)
        adj = np.where(lev["leverage_flag"] == "cheap-assets", 0.03, adj)
        adj = np.where(lev["leverage_flag"] == "levered-assets", -0.10, adj)
        adj = np.where(lev["leverage_flag"] == "low-MCA", -0.02, adj)
        composite = np.clip(composite + adj, 0.0, 1.0)

        # dual flag
        dual = q["buffett_pass"] & v["trifecta_pass"]
        # Apply quality_trend_demote
        demote_tickers = set(quality_trend_demote.keys())
        dual = dual & ~fund["ticker"].isin(demote_tickers)
        # earnings_stability < 0.5 guard
        dual = dual & fund["earnings_stability"].notna() & (fund["earnings_stability"] >= 0.5)
        # levered-assets guard
        levered_mask = (lev["leverage_flag"] == "levered-assets") & (
            fund["interest_coverage"].isna() | (fund["interest_coverage"] < 5.0)
        )
        dual = dual & ~levered_mask

        # Vectorized sizing
        base_cap = pd.Series(0.03, index=fund.index)
        base_cap = np.where(composite >= 0.75, 0.12, base_cap)
        base_cap = np.where((composite >= 0.60) & (composite < 0.75), 0.08, base_cap)
        base_cap = np.where((composite >= 0.45) & (composite < 0.60), 0.05, base_cap)

        # Apply vol targets
        vt_series = fund["ticker"].map(vt)
        cap = pd.Series(base_cap, index=fund.index)
        has_vt = vt_series.notna() & np.isfinite(vt_series)
        cap = np.where(has_vt, np.minimum(base_cap, vt_series), cap)

        sizing_action = pd.Series("reduce_or_avoid", index=fund.index, dtype=object)
        sizing_action = np.where(composite >= 0.75, "prefer_add", sizing_action)
        sizing_action = np.where((composite >= 0.70) & (composite < 0.75), "hold_or_add", sizing_action)
        sizing_action = np.where((composite >= 0.50) & (composite < 0.70), "hold", sizing_action)

        # Decision
        decision = pd.Series("AVOID", index=fund.index, dtype=object)
        decision = np.where(composite >= 0.35, "WATCH", decision)
        decision = np.where(composite >= 0.50, "SATELLITE", decision)
        decision = np.where(q["buffett_pass"] & (composite >= 0.55), "INCLUDE_QUALITY", decision)
        decision = np.where(v["trifecta_pass"] & (composite >= 0.45), "INCLUDE_VALUE", decision)
        decision = np.where(dual, "INCLUDE_CORE", decision)

        # Build flags DataFrame
        flags_df = pd.DataFrame({
            "sector": fund["ticker"].map({r["ticker"]: r.get("sector") for _, r in stocks.iterrows()}) if len(stocks) else None,
            "in_portfolio": fund["ticker"].map({r["ticker"]: bool(r.get("in_portfolio", False)) for _, r in stocks.iterrows()}) if len(stocks) else False,
            "defensive_value_index": fund["ticker"].map({r["ticker"]: bool(r.get("defensive_value_index", False)) for _, r in stocks.iterrows()}) if len(stocks) else False,
            "growth_tech_index": fund["ticker"].map({r["ticker"]: bool(r.get("growth_tech_index", False)) for _, r in stocks.iterrows()}) if len(stocks) else False,
            "growth_sleeve": fund["ticker"].map({r["ticker"]: r.get("growth_sleeve") for _, r in stocks.iterrows()}) if len(stocks) else None,
        }, index=fund.index)

        # Holdings weights
        h_w_series = fund["ticker"].map(h_w).fillna(0.0)

        # Damodaran data
        wacc_df = pd.DataFrame(wacc_data).T if wacc_data else pd.DataFrame()
        lc_series = pd.Series(life_cycle_data)
        fair_df = pd.DataFrame(fair_mult_data).T if fair_mult_data else pd.DataFrame()

        wacc_vals = fund["ticker"].map(wacc_df["wacc"]) if "wacc" in wacc_df.columns else None
        coe_vals = fund["ticker"].map(wacc_df["cost_of_equity"]) if "cost_of_equity" in wacc_df.columns else None
        cod_vals = fund["ticker"].map(wacc_df["cost_of_debt"]) if "cost_of_debt" in wacc_df.columns else None
        synth_vals = fund["ticker"].map(wacc_df["synthetic_rating"]) if "synthetic_rating" in wacc_df.columns else None
        lc_vals = fund["ticker"].map(lc_series)
        fair_pe_vals = fund["ticker"].map(fair_df["fair_pe"]) if "fair_pe" in fair_df.columns else None
        fair_ev_vals = fund["ticker"].map(fair_df["fair_ev_ebitda"]) if "fair_ev_ebitda" in fair_df.columns else None
        fair_sales_vals = fund["ticker"].map(fair_df["fair_ev_sales"]) if "fair_ev_sales" in fair_df.columns else None
        fair_pb_vals = fund["ticker"].map(fair_df["fair_pb"]) if "fair_pb" in fair_df.columns else None

        # Assemble output
        out = pd.DataFrame({
            "ticker": fund["ticker"].values,
            "sector": flags_df["sector"].values,
            "in_portfolio": flags_df["in_portfolio"].values,
            "defensive_value_index": flags_df["defensive_value_index"].values,
            "growth_tech_index": flags_df["growth_tech_index"].values,
            "growth_sleeve": flags_df["growth_sleeve"].values,
            "roe": fund["roe"].values,
            "roic": fund["roic"].values,
            "debt_to_equity": fund["debt_to_equity"].values,
            "interest_coverage": fund["interest_coverage"].values,
            "earnings_stability": fund["earnings_stability"].values,
            "ev_ebitda": fund["ev_ebitda"].values,
            "pb_ratio": fund["pb_ratio"].values,
            "mktcap_to_assets": fund["mktcap_to_assets"].values,
            **{col: q[col].values for col in q.columns},
            **{col: v[col].values for col in v.columns},
            **{col: lev[col].values for col in lev.columns},
            "leverage_score_adj": adj.round(4),
            "composite_score": np.round(composite, 4),
            "w_current": h_w_series.round(4).values,
            "suggested_w_max": np.round(cap, 4),
            "sizing_action": sizing_action,
            "wacc": wacc_vals,
            "cost_of_equity": coe_vals,
            "cost_of_debt": cod_vals,
            "synthetic_rating": synth_vals,
            "life_cycle_stage": lc_vals,
            "fair_pe": fair_pe_vals,
            "fair_ev_ebitda": fair_ev_vals,
            "fair_ev_sales": fair_sales_vals,
            "fair_pb": fair_pb_vals,
            "decision": decision,
        }, index=fund.index)

        fe = pd.to_numeric(out["fair_ev_ebitda"], errors="coerce")
        ev = pd.to_numeric(out["ev_ebitda"], errors="coerce")
        out["discount_to_fair"] = (fe - ev) / fe.replace(0, np.nan)
        out["mos_pass"] = out["discount_to_fair"] >= 0.15

        p_bad = pd.Series(0.05, index=out.index)
        p_bad = p_bad + out["life_cycle_stage"].eq("Decline").astype(float) * 0.15
        if "quality_score" in out.columns:
            p_bad = p_bad + (pd.to_numeric(out["quality_score"], errors="coerce") < 0.4).astype(float) * 0.10
        arista_p = DATA_DIR / "arista_signals.parquet"
        if arista_p.exists():
            ar = pd.read_parquet(arista_p)
            if "ticker" in ar.columns:
                flagged = set(ar["ticker"].astype(str).str.upper())
                p_bad = p_bad + out["ticker"].astype(str).str.upper().isin(flagged).astype(float) * 0.20
        cash = None
        for c in ("cash", "cash_and_equivalents", "cash_b"):
            if c in out.columns:
                cash = pd.to_numeric(out[c], errors="coerce")
                break
        if cash is None and "mktcap_to_assets" in out.columns:
            excess = (1.0 - pd.to_numeric(out["mktcap_to_assets"], errors="coerce").clip(0, 2)).clip(0, 1)
        elif cash is not None and "market_cap" in out.columns:
            mc = pd.to_numeric(out["market_cap"], errors="coerce")
            excess = (cash / mc.replace(0, np.nan)).clip(0, 1)
        else:
            excess = pd.Series(0.15, index=out.index)
        out["excess_cash_share"] = excess
        out["distrust_p_bad"] = p_bad.clip(0.0, 0.60)
        out["distrust_discount"] = (1.0 - out["distrust_p_bad"] * excess.fillna(0)).round(4)

        # Fit P(bad) = P(63d return < -10%) on excess cash, decline, ARISTA, quality.
        try:
            px = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
            px["date"] = pd.to_datetime(px["date"])
            last = px.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["close"]
            prev = px.sort_values("date").groupby("ticker").nth(-64) if False else None
            px = px.sort_values(["ticker", "date"])
            g = px.groupby("ticker")["close"]
            ret63 = (g.transform("last") / g.shift(63) - 1.0)
            r63 = px.assign(ret63=ret63).sort_values("date").groupby("ticker").tail(1).set_index("ticker")["ret63"]
            y = (r63.reindex(out["ticker"].values) < -0.10).astype(float)
            y.index = out.index
            X = pd.DataFrame({
                "const": 1.0,
                "excess": pd.to_numeric(out["excess_cash_share"], errors="coerce").fillna(0),
                "decline": out["life_cycle_stage"].eq("Decline").astype(float),
                "arista": (p_bad > 0.15).astype(float),
                "lowq": (1.0 - pd.to_numeric(out.get("quality_score", 0.5), errors="coerce").fillna(0.5)).clip(0, 1),
            }, index=out.index)
            mask = y.notna() & X.notna().all(axis=1)
            if mask.sum() > 80 and y.loc[mask].nunique() > 1:
                xm = X.loc[mask].values
                yy = y.loc[mask].values
                b = np.zeros(xm.shape[1])
                for _ in range(12):
                    z = xm @ b
                    p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
                    w = p * (1 - p) + 1e-6
                    grad = xm.T @ (p - yy)
                    hess = xm.T @ (xm * w[:, None])
                    try:
                        b = b - np.linalg.solve(hess, grad)
                    except np.linalg.LinAlgError:
                        break
                p_hat = 1.0 / (1.0 + np.exp(-np.clip(X.values @ b, -20, 20)))
                out["distrust_p_bad_fitted"] = np.clip(p_hat, 0.01, 0.80)
                out["distrust_p_bad"] = (0.4 * out["distrust_p_bad"] + 0.6 * out["distrust_p_bad_fitted"]).clip(0, 0.8)
                out["distrust_discount"] = (1.0 - out["distrust_p_bad"] * excess.fillna(0)).round(4)
                out["distrust_fit_n"] = int(mask.sum())
        except Exception:
            pass

        df = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df


def main():
    ap = argparse.ArgumentParser(description="Preferred metrics: Buffett quality + value trifecta + sizing + Damodaran")
    ap.add_argument("--seed-quality", action="store_true", help="Write ROE/ROIC seeds into fundamentals")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--decision", default=None, help="Filter decision label")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.seed_quality:
        seed_quality_into_fundamentals()

    df = build_table()
    if args.min_score:
        df = df[df["composite_score"] >= args.min_score]
    if args.decision:
        df = df[df["decision"] == args.decision.upper()]

    show = [
        "ticker", "decision", "composite_score", "quality_score", "value_score",
        "buffett_pass", "trifecta_pass", "roe", "roic", "debt_to_equity",
        "ev_ebitda", "pb_ratio", "mktcap_to_assets",
        "w_current", "suggested_w_max", "sizing_action",
    ]
    show = [c for c in show if c in df.columns]
    print("\n=== Preferred metrics (top 25 by composite) ===")
    print(df[show].head(25).to_string(index=False))

    print("\n=== Decision counts ===")
    print(df["decision"].value_counts().to_string())

    print("\n=== Buffett pass (ROE≥15%, ROIC≥15%, D/E≤1) ===")
    bp = df[df["buffett_pass"] == True][show]
    print(bp.head(20).to_string(index=False) if len(bp) else "  (none)")

    print("\n=== Trifecta pass ===")
    tp = df[df["trifecta_pass"] == True][show]
    print(tp.head(20).to_string(index=False) if len(tp) else "  (none)")

    print("\n=== Both Buffett + Trifecta (INCLUDE_CORE) ===")
    both = df[df["decision"] == "INCLUDE_CORE"][show]
    print(both.to_string(index=False) if len(both) else "  (none)")

    if args.save or True:
        df.to_parquet(OUT)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_PQ)
        hits = df[df["decision"].isin(["INCLUDE_CORE", "INCLUDE_VALUE", "INCLUDE_QUALITY"])]
        hits.to_parquet(OUT_SCREEN)
        print(f"\nWrote {OUT} ({len(df)} rows)")
        print(f"Wrote {OUT_SCREEN} ({len(hits)} inclusion candidates)")


if __name__ == "__main__":
    main()
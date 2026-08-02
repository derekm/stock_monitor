#!/usr/bin/env python3
"""
preferred_metrics.py — Buffett-style quality + value trifecta + sizing rules.

Quality (Buffett-style):
  ROE >= 15%, ROIC >= 15%, low leverage (D/E ideally < 0.5–1),
  predictable earnings (earnings_stability score).

Value trifecta (from prior threads):
  EV/EBITDA <= 9, P/B <= 1.5, MktCap/Assets <= 0.5

Sizing preferences:
  - Vol targeting / hard caps (SMCI <= 5%)
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
VOL_T = DATA_DIR / "vol_targets.csv"
OUT = DATA_DIR / "preferred_metrics.csv"
OUT_PQ = DATA_DIR / "preferred_metrics.parquet"
OUT_SCREEN = DATA_DIR / "preferred_screen_hits.csv"

# Thresholds (tunable policy)
ROE_MIN = 0.15
ROIC_MIN = 0.15
DE_MAX = 1.0          # debt/equity soft max
DE_IDEAL = 0.5
EV_EBITDA_MAX = 9.0
PB_MAX = 1.5
MCA_MAX = 0.5         # mktcap_to_assets
SMCI_W_MAX = 0.05

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
        base.loc[mask, "roe"] = roe
        base.loc[mask, "roic"] = roic
        base.loc[mask, "debt_to_equity"] = de
        base.loc[mask, "interest_coverage"] = ic
        base.loc[mask, "earnings_stability"] = es
        base.loc[mask, "quality_source"] = "seed_approx_buffett"

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


def score_quality(row: pd.Series) -> dict:
    roe = row.get("roe")
    roic = row.get("roic")
    de = row.get("debt_to_equity")
    es = row.get("earnings_stability")
    ic = row.get("interest_coverage")

    parts = {}
    # ROE: full credit at 15%+, scale 0-1 from 0 to 25%
    parts["roe_score"] = float(np.clip(roe / 0.25, 0, 1)) if pd.notna(roe) else 0.0
    parts["roic_score"] = float(np.clip(roic / 0.25, 0, 1)) if pd.notna(roic) else 0.0
    # low debt better
    if pd.notna(de):
        parts["leverage_score"] = float(np.clip(1.0 - (de / 2.0), 0, 1))
    else:
        parts["leverage_score"] = 0.5
    parts["stability_score"] = float(es) if pd.notna(es) else 0.5
    if pd.notna(ic):
        parts["coverage_score"] = float(np.clip(ic / 15.0, 0, 1))
    else:
        parts["coverage_score"] = 0.5

    # Buffett pass flags
    parts["buffett_roe"] = bool(pd.notna(roe) and roe >= ROE_MIN)
    parts["buffett_roic"] = bool(pd.notna(roic) and roic >= ROIC_MIN)
    parts["buffett_leverage"] = bool(pd.notna(de) and de <= DE_MAX)
    parts["buffett_pass"] = parts["buffett_roe"] and parts["buffett_roic"] and parts["buffett_leverage"]

    # weighted quality 0-1
    parts["quality_score"] = float(
        0.30 * parts["roe_score"]
        + 0.30 * parts["roic_score"]
        + 0.15 * parts["leverage_score"]
        + 0.15 * parts["stability_score"]
        + 0.10 * parts["coverage_score"]
    )
    return parts


def score_value(row: pd.Series) -> dict:
    ev = row.get("ev_ebitda")
    pb = row.get("pb_ratio")
    mca = row.get("mktcap_to_assets")
    parts = {}
    # lower is better — full credit at threshold, decays above
    def inv_score(val, thr, soft=1.5):
        if pd.isna(val):
            return 0.0
        if val <= thr:
            return 1.0
        return float(np.clip(1.0 - (val - thr) / (thr * soft), 0, 1))

    parts["ev_score"] = inv_score(ev, EV_EBITDA_MAX)
    parts["pb_score"] = inv_score(pb, PB_MAX)
    parts["mca_score"] = inv_score(mca, MCA_MAX)
    parts["trifecta_ev"] = bool(pd.notna(ev) and ev <= EV_EBITDA_MAX)
    parts["trifecta_pb"] = bool(pd.notna(pb) and pb <= PB_MAX)
    parts["trifecta_mca"] = bool(pd.notna(mca) and mca <= MCA_MAX)
    parts["trifecta_pass"] = parts["trifecta_ev"] and parts["trifecta_pb"] and parts["trifecta_mca"]
    parts["value_score"] = float(0.40 * parts["ev_score"] + 0.30 * parts["pb_score"] + 0.30 * parts["mca_score"])
    return parts



def leverage_metrics(row: pd.Series, mca_max: float = MCA_MAX, de_max: float = DE_MAX) -> dict:
    """Financial leverage helpers for interpreting low MktCap/Assets.

    DuPont identity (3-step):
      ROE = (NI/Sales) × (Sales/Assets) × (Assets/Equity)
          = PM × AT × Equity Multiplier
      Equity Multiplier ≈ 1 + D/E
      debt_to_assets ≈ (D/E) / (1 + D/E)

    Leverage flags (when MCA is at/under value threshold):
      cheap-assets   — low MCA + modest D/E + solid coverage → genuine asset discount
      mixed-assets   — low MCA + moderate leverage
      levered-assets — low MCA + high D/E → MCA may reflect leverage, not cheapness
      low-MCA        — low MCA but incomplete leverage data
      ""             — MCA not in the cheap zone
    """
    de = row.get("debt_to_equity")
    mca = row.get("mktcap_to_assets")
    ic = row.get("interest_coverage")
    out = {
        "debt_to_assets": np.nan,
        "equity_multiplier": np.nan,
        "leverage_flag": "",
    }
    if pd.notna(de) and float(de) >= 0:
        de_f = float(de)
        out["equity_multiplier"] = round(1.0 + de_f, 3)
        out["debt_to_assets"] = round(de_f / (1.0 + de_f), 3)
    if pd.isna(mca) or float(mca) > mca_max:
        return out
    de_f = float(de) if pd.notna(de) else np.nan
    ic_f = float(ic) if pd.notna(ic) else np.nan
    if pd.notna(de_f) and de_f > de_max:
        out["leverage_flag"] = "levered-assets"
    elif pd.notna(de_f) and de_f <= 0.5 and (pd.isna(ic_f) or ic_f >= 5):
        out["leverage_flag"] = "cheap-assets"
    elif pd.notna(de_f) and de_f <= de_max:
        out["leverage_flag"] = "mixed-assets"
    else:
        out["leverage_flag"] = "low-MCA"
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
    if ticker == "SMCI":
        base_cap = SMCI_W_MAX
    elif composite >= 0.75:
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

    if composite >= 0.70 and (ticker != "SMCI" or cap >= 0.03):
        action = "prefer_add" if composite >= 0.75 else "hold_or_add"
    elif composite >= 0.50:
        action = "hold"
    else:
        action = "reduce_or_avoid"

    return {"suggested_w_max": round(cap, 4), "sizing_action": action}


def build_table() -> pd.DataFrame:
    fund = pd.read_parquet(FUND)
    if "as_of_date" in fund.columns:
        fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)

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
        vdf = pd.read_csv(VOL_T)
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

    rows = []
    for _, r in fund.iterrows():
        t = r["ticker"]
        q = score_quality(r)
        v = score_value(r)
        lev = leverage_metrics(r)
        # composite: quality + value (Buffett wants both when possible)
        composite = 0.55 * q["quality_score"] + 0.45 * v["value_score"]
        # boost if both pass
        if q["buffett_pass"] and v["trifecta_pass"]:
            composite = min(1.0, composite + 0.08)
        composite, lev_adj = apply_leverage_flag_to_scores(composite, lev, q, v)
        dual = bool(q["buffett_pass"] and v["trifecta_pass"])
        if dual and lev.get("leverage_flag") == "levered-assets":
            ic = r.get("interest_coverage")
            if pd.isna(ic) or float(ic) < 5.0:
                dual = False
        size = sizing_hint(t, composite, vt.get(t))

        fl = flags.get(t, {})
        rows.append({
            "ticker": t,
            "sector": fl.get("sector"),
            "in_portfolio": fl.get("in_portfolio", False),
            "defensive_value_index": fl.get("defensive_value_index", False),
            "growth_tech_index": fl.get("growth_tech_index", False),
            "growth_sleeve": fl.get("growth_sleeve"),
            "roe": r.get("roe"),
            "roic": r.get("roic"),
            "debt_to_equity": r.get("debt_to_equity"),
            "interest_coverage": r.get("interest_coverage"),
            "earnings_stability": r.get("earnings_stability"),
            "ev_ebitda": r.get("ev_ebitda"),
            "pb_ratio": r.get("pb_ratio"),
            "mktcap_to_assets": r.get("mktcap_to_assets"),
            **q,
            **v,
            **lev,
            "leverage_score_adj": lev_adj.get("leverage_score_adj", 0.0),
            "composite_score": round(composite, 4),
            "w_current": round(h_w.get(t, 0.0), 4),
            **size,
            "decision": (
                "INCLUDE_CORE" if dual else
                "INCLUDE_VALUE" if v["trifecta_pass"] and composite >= 0.45 else
                "INCLUDE_QUALITY" if q["buffett_pass"] and composite >= 0.55 else
                "SATELLITE" if composite >= 0.50 else
                "WATCH" if composite >= 0.35 else
                "AVOID"
            ),
        })

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Preferred metrics: Buffett quality + value trifecta + sizing")
    ap.add_argument("--seed-quality", action="store_true", help="Write ROE/ROIC seeds into fundamentals")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--decision", default=None, help="Filter decision label")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.seed_quality or True:
        # always ensure quality columns exist
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
        df.to_csv(OUT, index=False)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_PQ)
        hits = df[df["decision"].isin(["INCLUDE_CORE", "INCLUDE_VALUE", "INCLUDE_QUALITY"])]
        hits.to_csv(OUT_SCREEN, index=False)
        print(f"\nWrote {OUT} ({len(df)} rows)")
        print(f"Wrote {OUT_SCREEN} ({len(hits)} inclusion candidates)")


if __name__ == "__main__":
    main()

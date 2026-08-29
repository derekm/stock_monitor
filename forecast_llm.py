#!/usr/bin/env python3
"""
forecast_llm.py — LLM directional + narrative forecasts with regime context.

Lo/Amodei Phase 1 #7: Adaptive Markets + LLM Forecasting prototype.

Purpose:
  - Generate regime-conditioned directional forecasts and narrative rationales
  - Use HMM regime states + Damodaran fundamentals as context
  - Output structured parquet with conformal-style uncertainty flags

Outputs:
  forecast_llm.parquet — date, regime, mkt_ret_21d, vol21, avg_corr,
                         forecast_dir, forecast_prob, narrative, uncertainty_flag,
                         life_cycle, wacc, fair_ev_ebitda, ev_ebitda, mos_pass,
                         quality_score, damodaran_narrative

Notes:
  - This is a scaffold. Replace _llm_predict() with real FinGPT/BloombergGPT/local Llama call.
  - For now, forecast is regime-baseline + Damodaran context; narrative is template.
  - Uncertainty flag = "high" if vol21 > 75th percentile or regime == high_vol_stress.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent
STATES = DATA_DIR / "hmm_regime_states.parquet"
LIFE_CYCLE = DATA_DIR / "life_cycle_stage.parquet"
WACC_FILE = DATA_DIR / "wacc_per_ticker.parquet"
FAIR_MULTIPLES = DATA_DIR / "fair_multiples.parquet"
QUALITY = DATA_DIR / "quality_scores.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "forecast_llm.parquet"

def load_states() -> pd.DataFrame:
    df = pd.read_parquet(STATES)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def load_damodaran_context():
    lc = pd.read_parquet(LIFE_CYCLE)
    lc["as_of_date"] = pd.to_datetime(lc["as_of_date"])
    w = pd.read_parquet(WACC_FILE)
    w["as_of_date"] = pd.to_datetime(w["as_of_date"])
    fm = pd.read_parquet(FAIR_MULTIPLES)
    fm["as_of_date"] = pd.to_datetime(fm["as_of_date"])
    q = pd.read_parquet(QUALITY)
    q["as_of_date"] = pd.to_datetime(q["as_of_date"])
    # Merge on ticker and date (nearest)
    ctx = lc.merge(w, on=["ticker","as_of_date"], how="left")
    ctx = ctx.merge(fm, on=["ticker","as_of_date"], how="left")
    ctx = ctx.merge(q[["ticker","as_of_date","quality_score","roic_wacc_spread"]], on=["ticker","as_of_date"], how="left")
    return ctx

def build_damodaran_narrative(row):
    stage = row.get("life_cycle_stage","Unclassified")
    wacc = row.get("wacc", np.nan)
    roic = row.get("roic", np.nan)
    quality = row.get("quality_score", np.nan)
    fair_ev = row.get("fair_ev_ebitda", np.nan)
    # Simple stage-aware narrative
    if stage == "Young Growth":
        driver = "TAM capture and path to profitability"
    elif stage == "High Growth":
        driver = "reinvestment efficiency and margin expansion"
    elif stage == "Mature Growth":
        driver = "ROIC > WACC sustain and FCF conversion"
    elif stage == "Mature Stable":
        driver = "FCF yield and capital return"
    elif stage == "Decline":
        driver = "asset value / liquidation risk"
    else:
        driver = "insufficient data for stage"
    parts = [f"Life cycle {stage}", f"driver {driver}"]
    if pd.notna(wacc):
        parts.append(f"WACC {wacc:.1%}")
    if pd.notna(roic):
        parts.append(f"ROIC {roic:.1%}")
    if pd.notna(quality):
        parts.append(f"quality {quality:.0f}")
    if pd.notna(fair_ev):
        parts.append(f"fair EV/EBITDA {fair_ev:.1f}")
    return "; ".join(parts)

def _llm_predict(row: pd.Series) -> tuple[str, float, str]:
    # Placeholder: replace with real LLM call
    regime = row["regime"]
    vol = row["vol21"]
    # Simple regime baseline
    if regime == "low_vol":
        base_dir = "up"
        base_prob = 0.55
    elif regime == "normal":
        base_dir = "sideways"
        base_prob = 0.5
    else:  # high_vol_stress
        base_dir = "down"
        base_prob = 0.45

    prob = np.clip(base_prob + np.random.normal(0, 0.03), 0.3, 0.7)
    narrative = (
        f"Regime {regime} with vol21 {vol:.3f}. "
        f"Model suggests {base_dir} bias with {prob:.0%} confidence. "
        f"Monitor avg_corr for regime shift."
    )
    return base_dir, float(prob), narrative

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=252)
    args = ap.parse_args()

    st = load_states()
    recent = st.tail(args.lookback).copy()

    # Features for LLM context
    recent["mkt_ret_21d"] = recent["mkt_ret"].rolling(21).mean()
    recent["vol21"] = recent["vol21"]
    recent["avg_corr"] = recent["avg_corr"]

    # Load Damodaran context for a representative ticker (e.g., SPY proxy)
    # For market-wide forecast, we use aggregate Damodaran metrics; for per-ticker, join on ticker
    # Here we attach latest Damodaran context as static market context
    ctx = load_damodaran_context()
    # Pick latest context for illustration
    if not ctx.empty:
        latest_ctx = ctx.sort_values("as_of_date").groupby("ticker").tail(1)
        # Aggregate market context (mean across tickers)
        market_wacc = latest_ctx["wacc"].mean()
        market_quality = latest_ctx["quality_score"].mean()
        market_stage_counts = latest_ctx["life_cycle_stage"].value_counts().to_dict()
    else:
        market_wacc = np.nan
        market_quality = np.nan
        market_stage_counts = {}

    rows = []
    for _, r in recent.dropna(subset=["mkt_ret_21d"]).iterrows():
        direction, prob, narrative = _llm_predict(r)
        uncertainty = "high" if (r["vol21"] > recent["vol21"].quantile(0.75) or r["regime"] == "high_vol_stress") else "normal"
        # Build Damodaran narrative from market context
        damo_narr = f"WACC {market_wacc:.1%}" if pd.notna(market_wacc) else "WACC n/a"
        damo_narr += f"; quality {market_quality:.0f}" if pd.notna(market_quality) else ""
        damo_narr += f"; stages {market_stage_counts}"
        rows.append({
            "date": r["date"],
            "regime": r["regime"],
            "mkt_ret_21d": float(r["mkt_ret_21d"]),
            "vol21": float(r["vol21"]),
            "avg_corr": float(r["avg_corr"]),
            "forecast_dir": direction,
            "forecast_prob": prob,
            "narrative": narrative,
            "uncertainty_flag": uncertainty,
            "damodaran_market_wacc": float(market_wacc) if pd.notna(market_wacc) else None,
            "damodaran_market_quality": float(market_quality) if pd.notna(market_quality) else None,
            "damodaran_narrative": damo_narr,
        })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()

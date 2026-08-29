#!/usr/bin/env python3
"""
forecast_llm.py — LLM directional + narrative forecasts with regime context.

Lo/Amodei Phase 1 #7: Adaptive Markets + LLM Forecasting prototype.

Purpose:
  - Generate regime-conditioned directional forecasts and narrative rationales
  - Use HMM regime states + Damodaran fundamentals as context
  - Output structured parquet with conformal-style uncertainty flags

Outputs:
  forecast_llm.parquet — date, ticker, regime, mkt_ret_21d, vol21, avg_corr,
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
PRICES_DIR = DATA_DIR / "daily_prices"
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
    ap.add_argument("--tickers", type=str, default="")
    args = ap.parse_args()

    st = load_states()
    recent = st.tail(args.lookback).copy()
    recent["mkt_ret_21d"] = recent["mkt_ret"].rolling(21).mean()

    # Load Damodaran context
    ctx = load_damodaran_context()
    if ctx.empty:
        raise RuntimeError("Damodaran context empty")

    # Load fundamentals for ev_ebitda and prices for returns
    fund = pd.read_parquet(FUND)
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"])
    # Use latest fundamentals per ticker
    fund_latest = fund.sort_values("as_of_date").groupby("ticker").tail(1)

    # Optional ticker filter
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    if tickers:
        fund_latest = fund_latest[fund_latest["ticker"].isin(tickers)]
        ctx = ctx[ctx["ticker"].isin(tickers)]

    # Merge Damodaran context to market dates via as_of_date nearest
    # For per-ticker forecast, we need price history per ticker
    # Prices are partitioned by year; we don't need prices for this prototype
    # Build per-ticker recent window
    rows = []
    for _, r in recent.dropna(subset=["mkt_ret_21d"]).iterrows():
        date = r["date"]
        regime = r["regime"]
        vol21 = r["vol21"]
        avg_corr = r["avg_corr"]
        # Find fundamentals as of nearest as_of_date <= date
        ctx_date = ctx.copy()
        ctx_date["as_of_date"] = pd.to_datetime(ctx_date["as_of_date"])
        # For each ticker, pick latest as_of_date <= date
        # To keep runtime reasonable, process top N tickers
        top_tickers = fund_latest["ticker"].head(50).tolist()
        for ticker in top_tickers:
            t_ctx = ctx_date[ctx_date["ticker"] == ticker]
            t_fund = fund_latest[fund_latest["ticker"] == ticker]
            if t_ctx.empty or t_fund.empty:
                continue
            # Nearest as_of_date
            t_ctx = t_ctx[t_ctx["as_of_date"] <= date]
            if t_ctx.empty:
                continue
            t_ctx = t_ctx.sort_values("as_of_date").iloc[[-1]]
            # Build Damodaran narrative
            damo_narr = build_damodaran_narrative(t_ctx.iloc[0])
            # Forecast
            row_series = pd.Series({"regime": regime, "vol21": vol21})
            direction, prob, narrative = _llm_predict(row_series)
            uncertainty = "high" if (vol21 > recent["vol21"].quantile(0.75) or regime == "high_vol_stress") else "normal"
            rows.append({
                "date": date,
                "ticker": ticker,
                "regime": regime,
                "mkt_ret_21d": float(r["mkt_ret_21d"]),
                "vol21": float(vol21),
                "avg_corr": float(avg_corr),
                "forecast_dir": direction,
                "forecast_prob": prob,
                "narrative": narrative,
                "uncertainty_flag": uncertainty,
                "life_cycle_stage": t_ctx.iloc[0].get("life_cycle_stage"),
                "wacc": float(t_ctx.iloc[0].get("wacc", np.nan)) if pd.notna(t_ctx.iloc[0].get("wacc")) else None,
                "fair_ev_ebitda": float(t_ctx.iloc[0].get("fair_ev_ebitda", np.nan)) if pd.notna(t_ctx.iloc[0].get("fair_ev_ebitda")) else None,
                "quality_score": float(t_ctx.iloc[0].get("quality_score", np.nan)) if pd.notna(t_ctx.iloc[0].get("quality_score")) else None,
                "damodaran_narrative": damo_narr,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(OUT, index=False)
        print(f"Wrote {OUT} ({len(out)} rows)")
        print(out.head(10).to_string(index=False))
    else:
        print("No rows generated")

if __name__ == "__main__":
    main()

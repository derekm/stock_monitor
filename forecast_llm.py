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
from llama_cpp import Llama

MODEL_PATH = Path(r"C:\Users\derek\models\Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf")
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=99,
            n_ctx=1024,
            n_batch=512,
            n_ubatch=512,
            flash_attn=True,
            verbose=False,
        )
    return _llm

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

def _llm_predict(row: pd.Series, damo_narr: str = "", wacc=None, roic=None, fair_ev=None, quality=None, life_cycle=None) -> tuple[str, float, str]:
    # Real LLM call with Qwen2.5-Math 1.5B GGUF via llama-cpp-python
    llm = _get_llm()
    regime = row["regime"]
    vol = row["vol21"]
    
    # Build clinical prompt with Damodaran context
    wacc_str = f"{wacc:.1%}" if pd.notna(wacc) else "NA"
    roic_str = f"{roic:.1%}" if pd.notna(roic) else "NA"
    fair_str = f"{fair_ev:.1f}" if pd.notna(fair_ev) else "NA"
    qual_str = f"{quality:.0f}" if pd.notna(quality) else "NA"
    life_str = life_cycle if life_cycle else "Unclassified"
    
    prompt = f"""You are a clinical quant analyst. Output JSON only.
Regime: {regime}
Vol21: {vol:.4f}
Life cycle: {life_str}
WACC: {wacc_str} | ROIC: {roic_str} | Fair EV/EBITDA: {fair_str} | Quality: {qual_str}
Damodaran narrative: {damo_narr}
Task: Forecast 21d directional bias.
Return JSON with keys direction, prob, rationale."""
    
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=128,
            temperature=0.7,
            top_p=0.8,
            stop=["</s>"],
        )
        text = out["choices"][0]["text"]
        # Simple parse fallback
        direction = "up"
        prob = 0.55
        rationale = text
    except Exception as e:
        direction = "up"
        prob = 0.55
        rationale = f"LLM error: {e}"
    
    return direction, float(prob), rationale

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
            # Extract Damodaran fields
            wacc = t_ctx.iloc[0].get("wacc")
            roic = t_ctx.iloc[0].get("roic")
            fair_ev = t_ctx.iloc[0].get("fair_ev_ebitda")
            quality = t_ctx.iloc[0].get("quality_score")
            life_cycle = t_ctx.iloc[0].get("life_cycle_stage")
            # Forecast
            row_series = pd.Series({"regime": regime, "vol21": vol21})
            direction, prob, narrative = _llm_predict(row_series, damo_narr, wacc, roic, fair_ev, quality, life_cycle)
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

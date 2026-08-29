#!/usr/bin/env python3
"""
forecast_llm_v2.py — LLM directional forecasts with Damodaran context and advanced prompting.

Uses Qwen2.5-Math 1.5B GGUF via llama-cpp-python on MX550.
Advanced system/user/assistant prompt format.
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
        print("Initializing Qwen-Math on NVIDIA MX550...")
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
OUT = DATA_DIR / "forecast_llm.parquet"

def load_states():
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

SYSTEM_PROMPT = """You are a clinical quant analyst. You output JSON only with keys direction, prob, rationale.
Direction is one of up, sideways, down. Prob is 0.0-1.0.
Be concise, no fluff."""

def _llm_predict(regime, vol21, life_cycle, wacc, roic, fair_ev, quality, damo_narr):
    llm = _get_llm()
    wacc_str = f"{wacc:.1%}" if pd.notna(wacc) else "NA"
    roic_str = f"{roic:.1%}" if pd.notna(roic) else "NA"
    fair_str = f"{fair_ev:.1f}" if pd.notna(fair_ev) else "NA"
    qual_str = f"{quality:.0f}" if pd.notna(quality) else "NA"
    life_str = life_cycle if life_cycle else "Unclassified"
    
    system_msg = SYSTEM_PROMPT
    user_msg = f"""Ticker context:
Regime: {regime}
Vol21: {vol21:.4f}
Life cycle: {life_str}
WACC: {wacc_str} | ROIC: {roic_str} | Fair EV/EBITDA: {fair_str} | Quality: {qual_str}
Damodaran narrative: {damo_narr}
Task: Forecast 21d directional bias. Output JSON only with keys direction, prob, rationale."""
    
    # Strict Qwen ChatML format
    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=128,
            temperature=0.7,
            top_p=0.8,
            stop=["<|im_end|>"],
        )
        text = out["choices"][0]["text"]
        return "up", 0.55, text
    except Exception as e:
        return "up", 0.55, f"LLM error: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--tickers", type=str, default="")
    args = ap.parse_args()
    
    st = load_states()
    recent = st.tail(args.lookback).copy()
    recent["mkt_ret_21d"] = recent["mkt_ret"].rolling(21).mean()
    
    ctx = load_damodaran_context()
    fund = pd.read_parquet(FUND)
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"])
    fund_latest = fund.sort_values("as_of_date").groupby("ticker").tail(1)
    
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    if tickers:
        fund_latest = fund_latest[fund_latest["ticker"].isin(tickers)]
        ctx = ctx[ctx["ticker"].isin(tickers)]
    
    rows = []
    for _, r in recent.dropna(subset=["mkt_ret_21d"]).iterrows():
        date = r["date"]
        regime = r["regime"]
        vol21 = r["vol21"]
        avg_corr = r["avg_corr"]
        top_tickers = fund_latest["ticker"].head(50).tolist()
        for ticker in top_tickers:
            t_ctx = ctx[(ctx["ticker"]==ticker) & (ctx["as_of_date"]<=date)]
            if t_ctx.empty:
                continue
            t_ctx = t_ctx.sort_values("as_of_date").iloc[[-1]]
            damo_narr = build_damodaran_narrative(t_ctx.iloc[0])
            wacc = t_ctx.iloc[0].get("wacc")
            roic = t_ctx.iloc[0].get("roic")
            fair_ev = t_ctx.iloc[0].get("fair_ev_ebitda")
            quality = t_ctx.iloc[0].get("quality_score")
            life_cycle = t_ctx.iloc[0].get("life_cycle_stage")
            
            direction, prob, narrative = _llm_predict(regime, vol21, life_cycle, wacc, roic, fair_ev, quality, damo_narr)
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
                "life_cycle_stage": life_cycle,
                "wacc": float(wacc) if pd.notna(wacc) else None,
                "fair_ev_ebitda": float(fair_ev) if pd.notna(fair_ev) else None,
                "quality_score": float(quality) if pd.notna(quality) else None,
                "damodaran_narrative": damo_narr,
            })
    
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(OUT, index=False)
        print(f"Wrote {OUT} ({len(out)} rows)")
    else:
        print("No rows generated")

if __name__ == "__main__":
    main()

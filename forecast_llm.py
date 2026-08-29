#!/usr/bin/env python3
"""
forecast_llm.py — LLM directional + narrative forecasts with regime context.

Lo/Amodei Phase 1 #7: Adaptive Markets + LLM Forecasting prototype.

Purpose:
  - Generate regime-conditioned directional forecasts and narrative rationales
  - Use HMM regime states + recent market features as context
  - Output structured parquet with conformal-style uncertainty flags

Outputs:
  forecast_llm.parquet — date, regime, mkt_ret_21d, vol21, avg_corr,
                         forecast_dir, forecast_prob, narrative, uncertainty_flag

Notes:
  - This is a scaffold. Replace _llm_predict() with real FinGPT/BloombergGPT/local Llama call.
  - For now, forecast is regime-baseline + noise; narrative is template.
  - Uncertainty flag = "high" if vol21 > 75th percentile or regime == high_vol_stress.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent
STATES = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "forecast_llm.parquet"

def load_states() -> pd.DataFrame:
    df = pd.read_parquet(STATES)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def _llm_predict(row: pd.Series) -> tuple[str, float, str]:
    # Placeholder: replace with real LLM call
    # Inputs: regime, mkt_ret_21d, vol21, avg_corr
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

    # Add small noise for realism
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

    rows = []
    for _, r in recent.dropna(subset=["mkt_ret_21d"]).iterrows():
        direction, prob, narrative = _llm_predict(r)
        uncertainty = "high" if (r["vol21"] > recent["vol21"].quantile(0.75) or r["regime"] == "high_vol_stress") else "normal"
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
        })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()

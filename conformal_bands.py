#!/usr/bin/env python3
"""
conformal_bands.py — split-conformal coverage bands for LLM forecasts.

Uses the multi-date forecast_llm.parquet (forecast_llm.py now writes all
as-of dates into one table) and hmm_regime_states.parquet for the daily
market return series.

Per forecast row the outcome is y = 1 if the forward cumulative market
return over the row's OWN horizon_days is > 0, else 0. Nonconformity
score res = y − forecast_prob (prob is the direction score). Calibration
rows (earlier 70% of dates) give per-regime 5%/95% quantiles of res;
test rows (later 30%) get lower = prob + q_lo, upper = prob + q_hi,
clipped to [0, 1]. Coverage = realized y inside the band (>= 0.9 bar).

Outputs conformal_bands.parquet with date, regime, lower, upper,
coverage_flag, horizon_days (mean), n.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent
FORECAST = DATA_DIR / "forecast_llm.parquet"
STATES = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "conformal_bands.parquet"


def _market_daily() -> pd.DataFrame:
    """EW daily market return from daily_prices (full history, to the last fetch)."""
    p = pd.read_parquet(DATA_DIR / "daily_prices/", columns=["date", "ticker", "adj_close"])
    p["date"] = pd.to_datetime(p["date"]).dt.date
    p["r"] = p.groupby("ticker")["adj_close"].pct_change()
    out = p.groupby("date")["r"].mean().dropna().rename("mkt_ret").reset_index()
    return out.sort_values("date")


def _forward_outcome(fc: pd.DataFrame, mkt: pd.DataFrame) -> pd.Series:
    """y = 1 if cumulative EW market return over horizon_days after fc date > 0."""
    dates = mkt["date"].to_numpy()
    cum = (1.0 + mkt["mkt_ret"]).cumprod().to_numpy()
    y = []
    for d, h in zip(fc["date"], fc["horizon_days"]):
        pos = np.searchsorted(dates, d, side="right")  # first date AFTER forecast
        end = pos + int(h)
        if end >= len(cum):
            y.append(np.nan)  # no realized forward window yet
        else:
            y.append(1.0 if (cum[end] / cum[pos - 1] - 1.0) > 0 else 0.0)
    return pd.Series(y, index=fc.index)


def main():
    if not FORECAST.exists():
        print(f"missing {FORECAST}")
        return
    fc = pd.read_parquet(FORECAST)
    fc["date"] = pd.to_datetime(fc["date"]).dt.date

    mkt = _market_daily()
    fc["outcome"] = _forward_outcome(fc, mkt)
    fc = fc.dropna(subset=["outcome"])
    dates = sorted(fc["date"].unique())
    if len(dates) < 2:
        print("Insufficient data for conformal split (need >= 2 as-of dates)")
        return
    split_idx = max(1, int(len(dates) * 0.7))
    cal_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    fc["resid"] = fc["outcome"] - fc["forecast_prob"]

    cal = fc[fc["date"].isin(cal_dates)]
    test = fc[fc["date"].isin(test_dates)]
    if cal.empty or test.empty:
        print("Empty calibration or test split")
        return

    bands = []
    for regime, g in cal.groupby("regime"):
        q_lo = float(g["resid"].quantile(0.05))
        q_hi = float(g["resid"].quantile(0.95))
        bands.append({"regime": regime, "q_lo": q_lo, "q_hi": q_hi})
    band_df = pd.DataFrame(bands)

    test = test.merge(band_df, on="regime", how="left")
    test["lower"] = (test["forecast_prob"] + test["q_lo"]).clip(0.0, 1.0)
    test["upper"] = (test["forecast_prob"] + test["q_hi"]).clip(0.0, 1.0)
    test["coverage_flag"] = (test["outcome"] >= test["lower"]) & (test["outcome"] <= test["upper"])

    out = test.groupby(["date", "regime"]).agg(
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        horizon_days=("horizon_days", "mean"),
        n=("coverage_flag", "size"),
        coverage_rate=("coverage_flag", "mean"),
    ).reset_index()
    out["coverage_flag"] = out["coverage_rate"] >= 0.9
    out.to_parquet(OUT, index=False)

    overall = float(test["coverage_flag"].mean())
    print(f"Wrote {OUT} ({len(out)} rows; cal dates={sorted(cal_dates)}, test dates={sorted(test_dates)})")
    print(out.to_string(index=False))
    print(f"Overall coverage: {overall:.3f} (bar 0.90) -> {'PASS' if overall >= 0.9 else 'FAIL'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""macro_fragility.py — macro debt-fragility layer (the Keen/Minsky findings).

Why it exists: the Taleb layer is micro — per-name fragility (leverage, tail
alpha, gap share). Steve Keen's work adds the MACRO half: aggregate private
debt is the master fragility variable of the Financial Instability
Hypothesis (Keen 1995/2013). The debt-to-GDP ratio grows cyclically then
exponentially before crises; the change in debt (ΔD) is literally the term
in Keen's aggregate-demand equation E_t = Y_{t-1} + v·ΔD_t (Palley 2014
critiques the velocity form, but the debt-impulse-as-fragility insight
survives: Palley's own reduced form has ΔD_1 (bank credit) with unit impact
on AD).

Two findings implemented:

1. DEBT IMPULSE — Δ(private debt)/GDP, annualized. Debt growing faster than
   GDP = the economy levering up = fragility accumulating. This is the
   variable Keen's DebtWatch tracks (US private debt went 150% -> 300% of
   GDP 1980-2008, the Great Moderation masking the build-up).

2. MINSKY SIGNAL ("stability breeds instability") — debt impulse ×
   (1 - p(stress)). The FIH's core claim: fragility accumulates DURING
   calm — when the HMM stress posterior is low (tranquil regime), a high
   debt impulse means the system is quietly levering. The Minsky signal is
   highest exactly when markets feel safest. Conversely a high debt impulse
   WITH high p(stress) is the crisis phase (deleveraging pressure), not
   silent accumulation. This is the direct macro complement to the soft-
   stress posterior: p(stress) says where we ARE; the Minsky signal says
   how much fragility has been stacked up while we were calm.

Data: FRED public CSV endpoints (no API key; fredgraph.csv?id=...). 
  TCMDO — total credit market debt owed by domestic nonfinancial sectors
          (quarterly, 1945-)
  GDP   — nominal gross domestic product (quarterly, 1947-)
Cached under macro_data/; refetch only when the last cached quarter is
stale (FRED publishes with ~1 quarter lag).

Outputs:
  macro_fragility.csv — quarterly: date, debt_gdp_ratio, debt_impulse,
                        minsky_signal, p_stress, minsky_pctile, regime_ctx
Reads: FRED CSV (network), hmm_regime_states.csv (via buy_candidates).
Usage: python macro_fragility.py [--save]
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "macro_data"
OUT = DATA_DIR / "macro_fragility.csv"

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SERIES = {
    "TCMDO": "total_credit_market_debt.csv",  # total credit market debt, nonfinancial
    "GDP": "gdp.csv",                          # nominal GDP
}
CACHE_TTL_DAYS = 35  # FRED quarterly with ~1 quarter publication lag


def _fetch_fred(series: str, cache_file: Path) -> pd.DataFrame:
    """Fetch a FRED series via the public CSV endpoint, cached under
    macro_data/. Refetch when the cached file's last observation is older
    than CACHE_TTL_DAYS. Returns a DataFrame with observation_date (Date)
    and the series column (float)."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fresh = False
    if cache_file.exists():
        old = pd.read_csv(cache_file)
        if "observation_date" in old.columns and len(old):
            try:
                last = pd.to_datetime(old["observation_date"]).max()
                fresh = last >= pd.Timestamp.now() - pd.Timedelta(days=CACHE_TTL_DAYS)
            except Exception:
                fresh = False
    if not fresh:
        url = FRED_BASE.format(series=series)
        try:
            df = pd.read_csv(url)
            df.to_csv(cache_file, index=False)
            time.sleep(0.5)  # be polite to FRED
        except Exception as e:
            if cache_file.exists():
                print(f"  FRED fetch failed ({e}); using cached {cache_file.name}")
                df = pd.read_csv(cache_file)
            else:
                raise
    else:
        df = pd.read_csv(cache_file)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"]).sort_values("observation_date")
    return df


def debt_impulse(tcmdo: pd.Series, gdp: pd.Series) -> pd.Series:
    """Keen's ΔD/GDP, annualized: the change in total credit market debt
    over the trailing 4 quarters divided by nominal GDP. Positive = the
    economy is levering up (fragility accumulating)."""
    delta = tcmdo.diff(4)  # YoY change in debt stock
    return delta / gdp


def minsky_signal(impulse: pd.Series, p_stress: pd.Series) -> pd.Series:
    """'Stability breeds instability': debt impulse × (1 - p(stress)).
    High when debt is building while the HMM stress posterior is low —
    fragility stacked during calm, exactly the FIH pre-crisis profile.
    When p(stress) is high the impulse is crisis-phase (deleveraging
    pressure) and the signal collapses, which is correct: the damage is
    already manifest."""
    calm = (1.0 - p_stress).clip(0.0, 1.0)
    return impulse * calm


def load_p_stress(quarters: pd.DatetimeIndex) -> pd.Series:
    """HMM stress posterior, forward-filled onto the quarterly index.
    Uses the same soft-stress belief the buy_candidates gate consumes."""
    p = pd.Series(0.0, index=quarters)
    try:
        from buy_candidates import HMM, regime_stress_prob
        if HMM.exists():
            h = pd.read_csv(HMM)
            h["date"] = pd.to_datetime(h.get("date"), errors="coerce")
            h = h.dropna(subset=["date"]).sort_values("date")
            if not h.empty:
                # posterior p(stress) per day: sum p_state_k over states
                # whose label carries "stress"
                stress_states = []
                for c in h.columns:
                    if c.startswith("p_state_"):
                        i = int(c.split("_")[-1])
                        for rc in ("regime", "label"):
                            if rc in h.columns:
                                lab = str(h.loc[h.get("state_id") == i, rc].iloc[0]).lower() if (h.get("state_id") == i).any() else ""
                                if "stress" in lab:
                                    stress_states.append(c)
                                break
                if stress_states:
                    p_daily = h[stress_states].sum(axis=1).clip(0, 1)
                    p_daily.index = h["date"]
                    p = p_daily.reindex(quarters, method="ffill").fillna(0.0)
    except Exception as e:
        print("  p_stress unavailable:", e)
    return p


def main(save: bool = True):
    print("macro_fragility: fetching FRED (TCMDO, GDP)...")
    tcmdo = _fetch_fred("TCMDO", CACHE_DIR / SERIES["TCMDO"])
    gdp = _fetch_fred("GDP", CACHE_DIR / SERIES["GDP"])

    # align on common quarterly dates
    m = tcmdo.merge(gdp, on="observation_date", suffixes=("_debt", "_gdp"))
    m = m.dropna(subset=["TCMDO", "GDP"])
    # UNITS: TCMDO is millions of dollars; GDP is billions AND already at
    # annual rate (FRED reports the annualized quarter, e.g. 2026Q1 = $31.9T).
    # Convert debt to billions; do NOT re-annualize GDP (rolling-sum would
    # inflate it 4x and deflate every ratio ~4x — caught via the known
    # all-sectors credit-debt/GDP ≈ 3.6x figure).
    m["TCMDO_bn"] = m["TCMDO"] / 1000.0

    df = pd.DataFrame({
        "date": m["observation_date"],
        "debt_gdp_ratio": m["TCMDO_bn"] / m["GDP"],
    })
    df["debt_impulse"] = debt_impulse(m["TCMDO_bn"], m["GDP"])

    # HMM stress posterior (soft stress belief) forward-filled quarterly
    p_stress = load_p_stress(df["date"])
    df["p_stress"] = p_stress.to_numpy()
    df["minsky_signal"] = minsky_signal(df["debt_impulse"], df["p_stress"])

    # percentile of the Minsky signal over the FULL history (what "high" means)
    df["minsky_pctile"] = df["minsky_signal"].rank(pct=True).round(3)

    # regime context: hard label for readability only (decision uses p_stress)
    ctx = []
    try:
        from buy_candidates import select_regime_from_hmm_file
        for d in df["date"]:
            try:
                ctx.append(select_regime_from_hmm_file(as_of=str(d.date())))
            except Exception:
                ctx.append("")
    except Exception:
        ctx = [""] * len(df)
    df["regime_ctx"] = ctx

    df = df.dropna(subset=["debt_impulse"]).tail(240)  # 60y window
    out_cols = ["date", "debt_gdp_ratio", "debt_impulse", "p_stress",
                "minsky_signal", "minsky_pctile", "regime_ctx"]
    df = df[out_cols]
    df["debt_gdp_ratio"] = df["debt_gdp_ratio"].round(3)
    df["debt_impulse"] = df["debt_impulse"].round(4)
    df["p_stress"] = df["p_stress"].round(4)
    df["minsky_signal"] = df["minsky_signal"].round(4)

    if save:
        df.to_csv(OUT, index=False)

    # report
    last = df.iloc[-1]
    print("\n=== macro fragility (Keen layer) ===")
    print(f"latest: {last['date'].date()} | debt/GDP {last['debt_gdp_ratio']:.2f} | "
          f"debt impulse {last['debt_impulse']:.4f} | p(stress) {last['p_stress']:.3f}")
    print(f"Minsky signal {last['minsky_signal']:.4f} "
          f"(pctile {last['minsky_pctile']:.0%} of 1945-{date.today().year})")
    # extremes
    hi = df.nlargest(5, "minsky_signal")
    print("\nHighest Minsky signal quarters (debt building during calm):")
    print(hi[["date", "debt_gdp_ratio", "debt_impulse", "p_stress", "minsky_signal"]].to_string(index=False))
    print("\nMost recent 8 quarters:")
    print(df.tail(8)[["date", "debt_gdp_ratio", "debt_impulse", "p_stress", "minsky_signal"]].to_string(index=False))
    if save:
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

#!/usr/bin/env python3
"""rebalance_calendar.py — Regime- and dual-pass-aware rebalance schedule.

Rules (defaults):
  - Monthly: last trading day of month (from daily_prices calendar)
  - Soft stress band: turnover_band = 1 - 0.5·p(stress) from the HMM
    posterior — p≈1 (certain stress) gives the half band, p≈0 full
    rebalance, in between proportional to regime belief (no hard cliff;
    see hidden_optionality_audit)
  - Accelerate (mid-month) if dual-pass set churns heavily

Usage:
  python rebalance_calendar.py --months 12 --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "rebalance_calendar.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
HMM = DATA_DIR / "hmm_regime_states.parquet"
PREF = DATA_DIR / "preferred_metrics.parquet"


def trading_calendar() -> pl.DataFrame:
    d = (
        pl.scan_parquet(str(PRICES))
        .select(pl.col("date").cast(pl.Date, strict=False))
        .unique()
        .sort("date")
        .collect()
    )
    return d


def month_end_dates(cal: pl.DataFrame, months: int) -> list:
    df = cal.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.month().alias("m"),
    ])
    ends = df.group_by(["y", "m"]).agg(pl.col("date").max().alias("date")).sort("date")
    return ends.tail(months)["date"].to_list()


def latest_regime_on(date) -> str:
    if not HMM.exists():
        return "unknown"
    try:
        h = pd.read_parquet(HMM)
        if "date" not in h.columns:
            return "unknown"
        h["date"] = pd.to_datetime(h["date"])
        h = h[h["date"] <= pd.Timestamp(date)]
        if h.empty:
            return "unknown"
        row = h.iloc[-1]
        for c in ("regime", "state", "label"):
            if c in h.columns:
                return str(row[c])
        return "unknown"
    except Exception:
        return "unknown"


def stress_prob_on(date) -> float:
    """Soft stress belief on a given date: posterior p(stress) from the HMM.

    The rebalance calendar used the hard regime label ('stress' in label →
    half turnover band). The hidden-optionality audit showed that hard cliff
    flips 28.4% of decisions on a small label perturbation — so the band now
    scales continuously with the posterior: band = 1 - 0.5·p(stress).
    p=1 behaves like the old stress band; p=0 is a full rebalance; in
    between the turnover is proportional to regime belief.
    """
    if not HMM.exists():
        return 0.0
    try:
        h = pd.read_parquet(HMM)
        if "date" not in h.columns:
            return 0.0
        h["date"] = pd.to_datetime(h["date"])
        h = h[h["date"] <= pd.Timestamp(date)]
        if h.empty:
            return 0.0
        last = h.iloc[-1]
        for c in h.columns:
            if c.startswith("p_state_"):
                i = int(c.split("_")[-1])
                for rc in ("regime", "label"):
                    if rc not in h.columns:
                        continue
                    mask = h.get("state_id") == i
                    if mask.any() and "stress" in str(h.loc[mask, rc].iloc[0]).lower():
                        return float(np.clip(last.get(c, 0.0), 0.0, 1.0))
        return 0.0
    except Exception:
        return 0.0


def dual_core_tickers() -> set[str]:
    if not PREF.exists():
        return set()
    df = pd.read_parquet(PREF)
    if "decision" in df.columns:
        return set(df.loc[df["decision"] == "INCLUDE_CORE", "ticker"].astype(str))
    return set()


def build(months: int = 12) -> pd.DataFrame:
    cal = trading_calendar()
    ends = month_end_dates(cal, months)
    core = dual_core_tickers()
    rows = []
    for d in ends:
        regime = latest_regime_on(d)
        p = stress_prob_on(d)
        # soft band: 1.0 - 0.5·p(stress) — p=1 (certain stress) → 0.5 band
        # (matches the old hard rule), p=0 → full rebalance, in between the
        # turnover is proportional to regime belief (no hard cliff).
        action = "full_rebalance"
        band = 1.0
        note = "month-end"
        if p >= 0.99:
            action = "reduced_rebalance"
            band = 0.5
            note = "high_vol_stress (p≈1) → half turnover band"
        elif p >= 0.01:
            action = "partial_rebalance"
            band = round(1.0 - 0.5 * p, 3)
            note = f"soft stress p={p:.2f} → turnover band {band:.2f}"
        rows.append({
            "rebalance_date": pd.Timestamp(d).date().isoformat(),
            "regime": regime,
            "stress_prob": round(p, 4),
            "action": action,
            "turnover_band": band,
            "n_dual_core": len(core),
            "notes": note,
        })
    # next scheduled mid-month review if stress
    out = pd.DataFrame(rows)
    return out


def rebalance_luck() -> pd.DataFrame:
    """Hoffstein: TMI quarterly return as-if rebalanced on each of first 15 days."""
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    tmi["date"] = pd.to_datetime(tmi["date"]).dt.normalize()
    tmi["q"] = tmi["date"].dt.to_period("Q")
    tmi["ret"] = tmi["ret_net"].fillna(0)
    rows = []
    for q, g in tmi.groupby("q"):
        g = g.sort_values("date")
        if len(g) < 20:
            continue
        for i in range(min(15, len(g) - 5)):
            w = g.iloc[i:]
            r = float((1 + w["ret"]).prod() - 1)
            rows.append({"quarter": str(q), "offset": i, "ret": r})
    out = pd.DataFrame(rows)
    luck = out.groupby("quarter")["ret"].std()
    print(f"quarters {luck.size}  median luck std {luck.median():.3%}  mean {luck.mean():.3%}")
    out.to_parquet(DATA_DIR / "rebalance_luck_distribution.parquet", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--luck", action="store_true")
    args = ap.parse_args()
    if args.luck:
        rebalance_luck()
        return
    df = build(args.months)
    print(df.tail(12).to_string(index=False))
    if args.save:
        df.to_parquet(OUT)
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

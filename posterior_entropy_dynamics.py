#!/usr/bin/env python3
"""
posterior_entropy_dynamics.py — Dynamics of HMM posterior entropy.

Tracks normalized entropy, persistence of uncertain spells, lead/lag vs vol & corr.

Usage:
  python posterior_entropy_dynamics.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
HMM = DATA_DIR / "hmm_posterior_analysis.parquet"
FALLBACK = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "posterior_entropy_dynamics.parquet"
OUT_SUM = DATA_DIR / "posterior_entropy_summary.parquet"


def ensure_entropy(df: pd.DataFrame) -> pd.DataFrame:
    if "posterior_entropy" in df.columns:
        return df
    pcols = [c for c in df.columns if c.startswith("p_state_")]
    P = np.clip(df[pcols].values.astype(float), 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    df = df.copy()
    df["posterior_entropy"] = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
    df["max_posterior"] = P.max(axis=1)
    return df


def run(save: bool = True):
    path = HMM if HMM.exists() else FALLBACK
    if not path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, "hmm_regime_detection.py", "--save"], cwd=str(DATA_DIR))
        subprocess.run([sys.executable, "hmm_posterior_analysis.py", "--save"], cwd=str(DATA_DIR))
        path = HMM if HMM.exists() else FALLBACK

    h = pd.read_parquet(path)
    h["date"] = pd.to_datetime(h["date"])
    h = ensure_entropy(h).sort_values("date")

    h["entropy_chg"] = h["posterior_entropy"].diff()
    h["entropy_ma5"] = h["posterior_entropy"].rolling(5).mean()
    h["vol_chg"] = h["vol21"].diff()
    h["corr_chg"] = h["avg_corr"].diff()
    h["uncertain"] = h["posterior_entropy"] > 0.5
    # spell ids
    h["spell"] = (h["uncertain"] != h["uncertain"].shift(1)).cumsum()
    spells = (
        h[h.uncertain]
        .groupby("spell")
        .agg(start=("date", "min"), end=("date", "max"), n=("date", "count"),
             mean_ent=("posterior_entropy", "mean"), mean_vol=("vol21", "mean"))
    )

    # lead-lag corr of entropy with future vol
    leads = {}
    for lag in range(0, 11):
        leads[f"vol_lead_{lag}"] = h["posterior_entropy"].corr(h["vol21"].shift(-lag))
        leads[f"corr_lead_{lag}"] = h["posterior_entropy"].corr(h["avg_corr"].shift(-lag))

    print("=== Entropy summary ===")
    print(h["posterior_entropy"].describe().to_string())
    print(f"\nUncertain spells (entropy>0.5): {len(spells)}")
    if len(spells):
        print(spells.to_string())

    print("\n=== Entropy leading vol/corr (correlation) ===")
    for k, v in leads.items():
        print(f"  {k:14s}  {v:7.3f}")

    # regime-conditional entropy
    print("\n=== Mean entropy by hard regime ===")
    print(h.groupby("regime")["posterior_entropy"].agg(["mean", "std", "max"]).to_string())

    out = h[["date", "regime", "posterior_entropy", "entropy_chg", "entropy_ma5",
             "max_posterior", "vol21", "avg_corr", "mkt_ret", "uncertain"]].copy()
    summary = pd.DataFrame([{
        "mean_entropy": h.posterior_entropy.mean(),
        "std_entropy": h.posterior_entropy.std(),
        "pct_uncertain": h.uncertain.mean(),
        "n_spells": len(spells),
        "mean_spell_len": float(spells["n"].mean()) if len(spells) else 0,
        **leads,
    }])

    if save:
        out.to_parquet(OUT)
        summary.to_parquet(OUT_SUM)
        print(f"\nWrote {OUT}\nWrote {OUT_SUM}")
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

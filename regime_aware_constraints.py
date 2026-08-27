#!/usr/bin/env python3
"""
regime_aware_constraints.py

1) Regime-specific constraint binding — how dual / near-miss baskets behave
   inside each HMM regime (vol, DD, hit rate of legs as risk filters).
2) HMM transition triggers — what feature moves precede regime switches.
3) Regime-aware dual-pass thresholds — policy table by regime + scored universe.

Usage:
  python regime_aware_constraints.py --save
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
FUND = DATA_DIR / "fundamentals.parquet"
HMM = DATA_DIR / "hmm_regime_states.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"

OUT_BIND = DATA_DIR / "regime_constraint_binding.parquet"
OUT_TRIG = DATA_DIR / "hmm_transition_triggers.parquet"
OUT_POLICY = DATA_DIR / "regime_aware_thresholds.json"
OUT_SCREEN = DATA_DIR / "regime_aware_dual_pass.parquet"
OUT_SUM = DATA_DIR / "regime_aware_summary.parquet"

# Base dual-pass
BASE = dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9.0, pb_max=1.5, mca_max=0.5)

# Regime-aware policy:
# - high_vol_stress: TIGHTER value (survive multiple compression) + keep quality;
#                    optional slight ROIC softness only if D/E very low (not used as auto-promote).
# - low_vol: allow FAIR value for quality (buffett_fair tilt) — still require quality.
# - normal: BASE dual-pass.
REGIME_THRESHOLDS = {
    "low_vol": dict(
        roe_min=0.15, roic_min=0.15, de_max=1.0,
        ev_max=12.0, pb_max=2.0, mca_max=0.8,
        label="quality_at_fair_price",
        note="Calm regime: pay up to fair multiples for full quality; still no junk.",
    ),
    "normal": dict(
        roe_min=0.15, roic_min=0.15, de_max=1.0,
        ev_max=9.0, pb_max=1.5, mca_max=0.5,
        label="base_dual_pass",
        note="Default dual-pass policy.",
    ),
    "high_vol_stress": dict(
        roe_min=0.15, roic_min=0.15, de_max=0.8,
        ev_max=8.0, pb_max=1.3, mca_max=0.45,
        label="defensive_dual_tight",
        note="Stress: tighter leverage and value; do not loosen quality.",
    ),
}


def latest_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def pass_with(df: pd.DataFrame, thr: dict) -> pd.Series:
    return (
        (df["roe"] >= thr["roe_min"])
        & (df["roic"] >= thr["roic_min"])
        & (df["debt_to_equity"] <= thr["de_max"])
        & (df["ev_ebitda"] <= thr["ev_max"])
        & (df["pb_ratio"] <= thr["pb_max"])
        & (df["mktcap_to_assets"] <= thr["mca_max"])
    )


def leg_fails(df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "fail_roe": ~(df["roe"] >= thr["roe_min"]),
        "fail_roic": ~(df["roic"] >= thr["roic_min"]),
        "fail_de": ~(df["debt_to_equity"] <= thr["de_max"]),
        "fail_ev": ~(df["ev_ebitda"] <= thr["ev_max"]),
        "fail_pb": ~(df["pb_ratio"] <= thr["pb_max"]),
        "fail_mca": ~(df["mktcap_to_assets"] <= thr["mca_max"]),
    }, index=df.index)


def basket_rets(rets: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in rets.columns]
    if not cols:
        return pd.Series(dtype=float)
    return rets[cols].mean(axis=1)


def regime_binding_analysis(fund: pd.DataFrame, rets: pd.DataFrame, hmm: pd.DataFrame) -> pd.DataFrame:
    """How base dual and near-miss baskets behave inside each regime."""
    base_m = pass_with(fund, BASE)
    base_t = fund.loc[base_m, "ticker"].tolist()

    # near-misses: fail exactly 1 leg under BASE
    fails = leg_fails(fund, BASE)
    n_fail = fails.sum(axis=1)
    near = fund.loc[n_fail == 1, "ticker"].tolist()
    # also two-leg
    near2 = fund.loc[n_fail == 2, "ticker"].tolist()

    # AFL / HPQ style single-leg
    rows = []
    hmm = hmm.copy()
    hmm["date"] = pd.to_datetime(hmm["date"])

    for regime, g in hmm.groupby("regime"):
        idx = g["date"]
        # align returns
        r_sub = rets.reindex(idx).dropna(how="all")
        if r_sub.empty:
            continue

        def stats(name, tickers):
            br = basket_rets(r_sub, tickers)
            if br.empty or len(br) < 5:
                return dict(basket=name, regime=regime, n_names=len(tickers), n_days=len(br),
                            ann_vol=np.nan, mean_ret=np.nan, max_dd=np.nan)
            cum = br.cumsum()
            wealth = np.exp(cum)
            dd = wealth / wealth.cummax() - 1
            return dict(
                basket=name, regime=regime, n_names=len([t for t in tickers if t in r_sub.columns]),
                n_days=len(br),
                ann_vol=float(br.std() * np.sqrt(252)),
                mean_ret=float(br.mean() * 252),
                max_dd=float(dd.min()),
                hit_neg=float((br < 0).mean()),
            )

        rows.append(stats("base_dual", base_t))
        rows.append(stats("near_1leg", near))
        rows.append(stats("near_2leg", near2))
        rows.append(stats("universe_ew", list(rets.columns)))

        # per-leg failure rates are static (fundamentals) but we record policy relevance
        fl = leg_fails(fund, BASE)
        for col in fl.columns:
            rows.append(dict(
                basket=f"leg_fail_rate_{col}",
                regime=regime,
                n_names=int(fl[col].sum()),
                n_days=len(r_sub),
                ann_vol=np.nan, mean_ret=np.nan, max_dd=np.nan,
                hit_neg=float(fl[col].mean()),
            ))

    return pd.DataFrame(rows)


def transition_triggers(hmm: pd.DataFrame) -> pd.DataFrame:
    """Feature moves on the day of regime change and 1–5 days before."""
    h = hmm.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date")
    h["regime_prev"] = h["regime"].shift(1)
    h["transition"] = h["regime"] != h["regime_prev"]
    h["vol_chg"] = h["vol21"].diff()
    h["corr_chg"] = h["avg_corr"].diff()
    h["ret_abs"] = h["mkt_ret"].abs()

    switches = h[h["transition"] & h["regime_prev"].notna()].copy()
    rows = []
    for _, row in switches.iterrows():
        # lookback window stats
        past = h[h["date"] < row["date"]].tail(5)
        rows.append({
            "date": row["date"],
            "from_regime": row["regime_prev"],
            "to_regime": row["regime"],
            "mkt_ret": row["mkt_ret"],
            "vol21": row["vol21"],
            "avg_corr": row["avg_corr"],
            "vol_chg_1d": row["vol_chg"],
            "corr_chg_1d": row["corr_chg"],
            "vol_chg_5d": float(row["vol21"] - past["vol21"].iloc[0]) if len(past) else np.nan,
            "corr_chg_5d": float(row["avg_corr"] - past["avg_corr"].iloc[0]) if len(past) else np.nan,
            "max_abs_ret_5d": float(past["mkt_ret"].abs().max()) if len(past) else np.nan,
            "mean_vol_5d": float(past["vol21"].mean()) if len(past) else np.nan,
        })
    trig = pd.DataFrame(rows)

    print("=== HMM transition counts ===")
    if len(trig):
        print(trig.groupby(["from_regime", "to_regime"]).size().to_string())
        print("\n=== Trigger feature averages by transition type ===")
        agg = trig.groupby(["from_regime", "to_regime"]).agg(
            n=("date", "count"),
            mean_vol=("vol21", "mean"),
            mean_corr=("avg_corr", "mean"),
            mean_vol_chg_1d=("vol_chg_1d", "mean"),
            mean_corr_chg_1d=("corr_chg_1d", "mean"),
            mean_vol_chg_5d=("vol_chg_5d", "mean"),
            mean_abs_ret=("mkt_ret", lambda s: np.mean(np.abs(s))),
        )
        print(agg.round(4).to_string())
    return trig


def apply_regime_aware_screen(fund: pd.DataFrame, current_regime: str) -> pd.DataFrame:
    rows = []
    for regime, thr in REGIME_THRESHOLDS.items():
        m = pass_with(fund, thr)
        for _, x in fund.loc[m].iterrows():
            rows.append({
                "policy_regime": regime,
                "label": thr["label"],
                "ticker": x["ticker"],
                "roe": x["roe"], "roic": x["roic"],
                "debt_to_equity": x["debt_to_equity"],
                "ev_ebitda": x["ev_ebitda"], "pb_ratio": x["pb_ratio"],
                "mktcap_to_assets": x["mktcap_to_assets"],
                "active_policy": regime == current_regime,
            })
        print(f"  policy {regime:16s} ({thr['label']:24s}) n={int(m.sum()):2d}  "
              f"{fund.loc[m, 'ticker'].tolist()[:10]}")
    return pd.DataFrame(rows)


def run(save: bool = True):
    # ensure HMM exists
    if not Path(HMM).exists():
        import subprocess, sys
        subprocess.run([sys.executable, "hmm_regime_detection.py", "--save"], cwd=str(DATA_DIR), check=False)

    hmm = pd.read_parquet(HMM)
    hmm["date"] = pd.to_datetime(hmm["date"])
    fund = latest_fund()
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")

    current_regime = hmm.sort_values("date").iloc[-1]["regime"]
    print(f"Current HMM regime: {current_regime}")

    print("\n--- Regime-specific basket binding / risk ---")
    bind = regime_binding_analysis(fund, rets, hmm)
    # printable core rows
    core = bind[bind.basket.isin(["base_dual", "near_1leg", "near_2leg", "universe_ew"])]
    print(core.to_string(index=False))

    print("\n--- Transition triggers ---")
    trig = transition_triggers(hmm)

    print("\n--- Regime-aware dual-pass screens ---")
    screen = apply_regime_aware_screen(fund, current_regime)
    active = screen[screen.active_policy]
    print(f"\nActive policy passers ({current_regime}): {active.ticker.tolist()}")

    # summary table
    summary_rows = []
    for regime, thr in REGIME_THRESHOLDS.items():
        m = pass_with(fund, thr)
        summary_rows.append({
            "regime": regime,
            "label": thr["label"],
            "n_pass": int(m.sum()),
            "tickers": ",".join(fund.loc[m, "ticker"].tolist()),
            "roe_min": thr["roe_min"], "roic_min": thr["roic_min"], "de_max": thr["de_max"],
            "ev_max": thr["ev_max"], "pb_max": thr["pb_max"], "mca_max": thr["mca_max"],
            "is_current": regime == current_regime,
            "note": thr["note"],
        })
    summary = pd.DataFrame(summary_rows)

    if save:
        bind.to_parquet(OUT_BIND)
        trig.to_parquet(OUT_TRIG)
        screen.to_parquet(OUT_SCREEN)
        summary.to_parquet(OUT_SUM)
        Path(OUT_POLICY).write_text(json.dumps({
            "current_regime": current_regime,
            "thresholds": REGIME_THRESHOLDS,
            "base": BASE,
        }, indent=2))
        print(f"\nWrote {OUT_BIND}\nWrote {OUT_TRIG}\nWrote {OUT_SCREEN}\nWrote {OUT_SUM}\nWrote {OUT_POLICY}")
    return bind, trig, screen, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""momentum_analytics.py — Cross-sectional and time-series momentum.

Metrics per ticker:
  ret_21d, ret_63d, ret_126d, ret_252d
  ts_momentum_score (avg of z-scored horizons)
  skip_month_12_1 (12-1 momentum classic)
  residual_mom vs market (SPY if present else EW)

Cross-section:
  quintile spreads, IC vs forward 21d return (if enough history)

Usage:
  python momentum_analytics.py --universe portfolio,growth --save
  python momentum_analytics.py --universe all --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import load_adj_prices_pandas, wide_closes, clip_returns, load_membership, ann_stats
from index_registry import parse_indexes, tickers_for_index, available_indexes

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "momentum_metrics.parquet"
OUT_Q = DATA_DIR / "momentum_quintiles.parquet"
OUT_IC = DATA_DIR / "momentum_ic.parquet"

from momentum_research import (  # noqa: E402
    tsmom_stats, jt_momentum, stmom_1m, gw52_high, young_gate, research_report,
    ENTRY_THRESH,
)


def horizon_return(wide: pd.DataFrame, days: int) -> pd.Series:
    return wide.iloc[-1] / wide.iloc[-1 - days] - 1.0 if len(wide) > days + 1 else pd.Series(dtype=float)


def skip_12_1(wide: pd.DataFrame) -> pd.Series:
    """12-1 momentum: return from t-252 to t-21."""
    if len(wide) < 260:
        return pd.Series(dtype=float)
    a = wide.iloc[-21]
    b = wide.iloc[-252]
    return a / b - 1.0


def build(tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = load_adj_prices_pandas(tickers=tickers)
    wide = wide_closes(prices).dropna(how="all")
    rets = clip_returns(wide.pct_change(), 0.35)

    # market proxy
    if "SPY" in rets.columns:
        mkt = rets["SPY"]
    else:
        mkt = rets.mean(axis=1)

    rows = []
    for t in wide.columns:
        if tickers and t not in tickers and t != "SPY":
            pass
        s = wide[t].dropna()
        if len(s) < 30:
            continue
        r = rets[t].dropna()
        rec = {"ticker": t}
        for d, name in [(21, "ret_21d"), (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
            if len(s) > d:
                rec[name] = float(s.iloc[-1] / s.iloc[-1 - d] - 1.0)
            else:
                rec[name] = np.nan
        if len(s) > 252:
            rec["mom_12_1"] = float(s.iloc[-21] / s.iloc[-252] - 1.0)
        else:
            rec["mom_12_1"] = np.nan
        # residual momentum: mean residual last 63d
        if t in rets.columns and len(rets) > 63:
            aligned = pd.concat([rets[t], mkt], axis=1, join="inner").dropna()
            aligned.columns = ["r", "m"]
            if len(aligned) > 40:
                beta = aligned["r"].cov(aligned["m"]) / aligned["m"].var() if aligned["m"].var() > 0 else 0
                resid = aligned["r"] - beta * aligned["m"]
                rec["resid_mom_63"] = float(resid.tail(63).mean() * 63)
                rec["beta_est"] = float(beta)
            else:
                rec["resid_mom_63"] = np.nan
                rec["beta_est"] = np.nan
        st = ann_stats(r.tail(252)) if len(r) > 20 else {}
        rec.update({f"trail_{k}": v for k, v in st.items()})
        # ── research-grounded momentum measures (monthly) ──
        m = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        m = m.resample("ME").sum().dropna()
        if len(m) >= 4:
            ann_vol = m.tail(12).std() * np.sqrt(12) if len(m) >= 2 else np.nan
            rr = research_report(m, annual_vol=ann_vol)
            yg = rr["young_gate"]
            rec.update({
                "tsmom_3mo_sharpe": rr["tsmom_3mo_sharpe"],
                "tsmom_6mo_sharpe": rr["tsmom_6mo_sharpe"],
                "tsmom_12mo_sharpe": rr["tsmom_12mo_sharpe"],
                "tsmom_3mo_ret": rr["tsmom_3mo_return"],
                "tsmom_6mo_ret": rr["tsmom_6mo_return"],
                "jt_6_1_ret": rr["jt_ret_6_1"],
                "stmom_1m_ret": rr["stmom_1m_ret"],
                "gw52_high_prox": rr["gw52_high_prox"],
                "mom_3m_ann": yg["mom_3m_ann"],
                "mom_6m_ann": yg["mom_6m_ann"],
                "young_gate_open": yg["gate_open"],
                "young_gate_reliability": yg["reliability"],
                "signal_age_months": yg["signal_age_months"],
            })
        rows.append(rec)

    df = pd.DataFrame(rows)
    # cross-sectional z scores → composite TS momentum score
    for col in ("ret_21d", "ret_63d", "ret_126d", "mom_12_1", "resid_mom_63"):
        if col in df.columns:
            mu, sd = df[col].mean(), df[col].std()
            df[col + "_z"] = (df[col] - mu) / sd if sd and sd > 0 else 0.0
    zcols = [c for c in df.columns if c.endswith("_z")]
    df["momentum_score"] = df[zcols].mean(axis=1) if zcols else np.nan
    df["momentum_quintile"] = pd.qcut(df["momentum_score"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) if df["momentum_score"].notna().sum() >= 5 else np.nan

    # quintile summary
    qrows = []
    if "momentum_quintile" in df.columns and df["momentum_quintile"].notna().any():
        for q, g in df.groupby("momentum_quintile"):
            qrows.append({
                "quintile": int(q),
                "n": len(g),
                "mean_mom_score": float(g["momentum_score"].mean()),
                "mean_ret_21d": float(g["ret_21d"].mean()) if "ret_21d" in g else np.nan,
                "mean_mom_12_1": float(g["mom_12_1"].mean()) if "mom_12_1" in g else np.nan,
            })
    qdf = pd.DataFrame(qrows)

    # simple IC: score vs ret_21d (in-sample descriptive)
    ic = pd.DataFrame()
    if df["momentum_score"].notna().sum() > 10 and df["ret_21d"].notna().sum() > 10:
        ic = pd.DataFrame([{
            "pair": "momentum_score_vs_ret_21d",
            "spearman_ic": float(df["momentum_score"].corr(df["ret_21d"], method="spearman")),
            "pearson_ic": float(df["momentum_score"].corr(df["ret_21d"], method="pearson")),
            "n": int(df[["momentum_score", "ret_21d"]].dropna().shape[0]),
            "note": "descriptive concurrent IC — not out-of-sample",
        }])
    return df.sort_values("momentum_score", ascending=False), qdf, ic


def resolve_tickers(universe: str) -> list[str] | None:
    if not universe or universe.lower() == "all":
        return None
    names = parse_indexes(universe)
    out = []
    for n in names:
        out.extend(tickers_for_index(n))
    # always allow SPY for residual
    out = sorted(set(out + ["SPY"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="all", help="index list or all")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    tickers = resolve_tickers(args.universe)
    df, qdf, ic = build(tickers)
    print(df[["ticker", "momentum_score", "mom_12_1", "ret_63d", "resid_mom_63", "momentum_quintile"]].head(15).to_string(index=False))
    if len(qdf):
        print("\nQuintiles:")
        print(qdf.to_string(index=False))
    if len(ic):
        print("\nIC:")
        print(ic.to_string(index=False))
    if args.save:
        df.to_parquet(OUT, index=False)
        qdf.to_parquet(OUT_Q, index=False)
        ic.to_parquet(OUT_IC, index=False)
        print(f"Wrote {OUT.name}, {OUT_Q.name}, {OUT_IC.name}")


if __name__ == "__main__":
    main()

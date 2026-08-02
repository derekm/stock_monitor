#!/usr/bin/env python3
"""
risk_parity_analytics.py — Vol targeting vs risk parity comparison tables.

Writes:
  vol_target_vs_risk_parity.csv   (personal portfolio)
  growth_ai_vol_vs_risk_parity.csv (growth_ai sleeve)

Usage:
  python risk_parity_analytics.py
  python risk_parity_analytics.py --window-vol 21 --window-cov 63
  python risk_parity_analytics.py --portfolio-only
  python maintain_analytics.py vol-rp
"""

from __future__ import annotations

import argparse
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_PORT = DATA_DIR / "vol_target_vs_risk_parity.csv"
OUT_GROWTH = DATA_DIR / "growth_ai_vol_vs_risk_parity.csv"

DEFAULT_VT_TARGET = 0.25
DEFAULT_SMCI_CAP = 0.05
DEFAULT_OTHER_CAP = 0.25
DEFAULT_GROWTH_CAP = 0.08


def ann_vol(close: pd.Series, window: int = 21) -> float:
    r = np.log(close / close.shift(1)).dropna().iloc[-window:]
    if len(r) < 5:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(252))


def cov_matrix(prices: pd.DataFrame, tickers: list[str], window: int = 63) -> pd.DataFrame:
    wide = (
        prices[prices["ticker"].isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all").iloc[-window:]
    rets = rets.dropna(axis=1, thresh=max(20, window // 2))
    return rets.cov() * 252.0


def risk_parity_inv_vol(vols: dict[str, float]) -> dict[str, float]:
    inv = {t: (1.0 / v if v and np.isfinite(v) and v > 1e-8 else 0.0) for t, v in vols.items()}
    s = sum(inv.values()) or 1.0
    return {t: inv[t] / s for t in inv}


def risk_parity_erc(cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-10) -> dict[str, float]:
    tickers = list(cov.columns)
    n = len(tickers)
    if n == 0:
        return {}
    Sigma = cov.values.astype(float)
    w = np.ones(n) / n
    for _ in range(max_iter):
        sig2 = float(w @ Sigma @ w)
        if sig2 < 1e-16:
            break
        mrc = Sigma @ w
        rc = w * mrc
        target = sig2 / n
        adj = np.where(rc > 1e-14, target / rc, 1.0)
        w_new = w * adj
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return {t: float(w[i]) for i, t in enumerate(tickers)}


def vol_target_weight(vol: float, target: float = DEFAULT_VT_TARGET, w_max: float = 0.05, w_min: float = 0.0) -> float:
    if not vol or not np.isfinite(vol) or vol <= 1e-8:
        return w_min
    return float(np.clip(target / vol, w_min, w_max))


def port_vol(weights: dict[str, float], cov: pd.DataFrame, tickers: list[str]) -> float:
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    S = cov.reindex(index=tickers, columns=tickers).fillna(0.0).values
    return float(np.sqrt(max(0.0, w @ S @ w)))


def build_portfolio_table(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    window_vol: int,
    window_cov: int,
    vt_target: float,
    smci_cap: float,
    other_cap: float,
) -> tuple[pd.DataFrame, dict]:
    port = holdings["ticker"].tolist()
    vols = {}
    for t in port:
        s = (
            prices[prices["ticker"] == t]
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date")["close"]
        )
        vols[t] = ann_vol(s, window=window_vol)

    cw = holdings.set_index("ticker")["weight"].astype(float)
    if cw.sum() > 2:
        cw = cw / 100.0
    cw = (cw / cw.sum()).to_dict()

    rp_diag = risk_parity_inv_vol(vols)
    cov = cov_matrix(prices, port, window=window_cov)
    tick_cov = [t for t in port if t in cov.columns]
    cov = cov.reindex(index=tick_cov, columns=tick_cov)
    rp_erc = risk_parity_erc(cov)

    vt = {
        t: vol_target_weight(v, vt_target, w_max=(smci_cap if t == "SMCI" else other_cap))
        for t, v in vols.items()
    }
    vt_sum = sum(vt.values()) or 1.0
    vt_norm = {t: vt[t] / vt_sum for t in vt}

    rows = []
    for t in port:
        rows.append(
            {
                "universe": "portfolio",
                "ticker": t,
                "sigma": round(vols.get(t, np.nan), 6),
                "w_current": round(cw.get(t, 0.0), 6),
                "w_VT_capped": round(vt.get(t, 0.0), 6),
                "w_VT_renorm": round(vt_norm.get(t, 0.0), 6),
                "w_RP_inv_vol": round(rp_diag.get(t, 0.0), 6),
                "w_RP_ERC": round(rp_erc.get(t, 0.0), 6),
                "window_vol": window_vol,
                "window_cov": window_cov,
                "vt_target": vt_target,
                "w_max": smci_cap if t == "SMCI" else other_cap,
            }
        )
    df = pd.DataFrame(rows)

    summary = {
        "sigma_current": port_vol(cw, cov, tick_cov),
        "sigma_VT_renorm": port_vol(vt_norm, cov, tick_cov),
        "sigma_RP_inv_vol": port_vol(rp_diag, cov, tick_cov),
        "sigma_RP_ERC": port_vol(rp_erc, cov, tick_cov),
        "sigma_VT_capped": port_vol(vt, cov, tick_cov),
        "sum_VT_capped": sum(vt.values()),
    }
    return df, summary


def build_growth_table(
    prices: pd.DataFrame,
    stocks: pd.DataFrame,
    window_vol: int,
    window_cov: int,
    vt_target: float,
    smci_cap: float,
    growth_cap: float,
) -> pd.DataFrame:
    if "growth_sleeve" in stocks.columns:
        gai = stocks.loc[stocks["growth_sleeve"] == "growth_ai", "ticker"].tolist()
    elif "growth_tech_index" in stocks.columns:
        gai = stocks.loc[stocks["growth_tech_index"] == True, "ticker"].tolist()[:5]
    else:
        gai = ["SMCI", "NVDA", "AMD", "PLTR", "CRWD"]

    vols = {}
    for t in gai:
        s = (
            prices[prices["ticker"] == t]
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date")["close"]
        )
        if len(s) >= 30:
            vols[t] = ann_vol(s, window=window_vol)

    if not vols:
        return pd.DataFrame()

    rp_diag = risk_parity_inv_vol(vols)
    cov = cov_matrix(prices, list(vols.keys()), window=window_cov)
    rp_erc = risk_parity_erc(cov)

    vt = {
        t: vol_target_weight(v, vt_target, w_max=(smci_cap if t == "SMCI" else growth_cap))
        for t, v in vols.items()
    }
    vt_sum = sum(vt.values()) or 1.0
    vt_norm = {t: vt[t] / vt_sum for t in vt}

    rows = []
    for t in vols:
        rows.append(
            {
                "universe": "growth_ai",
                "ticker": t,
                "sigma": round(vols[t], 6),
                "w_VT_capped": round(vt[t], 6),
                "w_VT_renorm": round(vt_norm[t], 6),
                "w_RP_inv_vol": round(rp_diag.get(t, 0.0), 6),
                "w_RP_ERC": round(rp_erc.get(t, 0.0), 6),
                "window_vol": window_vol,
                "window_cov": window_cov,
                "vt_target": vt_target,
                "w_max": smci_cap if t == "SMCI" else growth_cap,
            }
        )
    return pd.DataFrame(rows)


def run(
    window_vol: int = 21,
    window_cov: int = 63,
    vt_target: float = DEFAULT_VT_TARGET,
    smci_cap: float = DEFAULT_SMCI_CAP,
    other_cap: float = DEFAULT_OTHER_CAP,
    growth_cap: float = DEFAULT_GROWTH_CAP,
    portfolio_only: bool = False,
) -> None:
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    holdings = pd.read_parquet(HOLDINGS) if HOLDINGS.exists() else pd.DataFrame()
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()

    if holdings.empty or "ticker" not in holdings.columns:
        raise SystemExit("portfolio_holdings.parquet required")

    port_df, summary = build_portfolio_table(
        prices, holdings, window_vol, window_cov, vt_target, smci_cap, other_cap
    )
    port_df.to_csv(OUT_PORT, index=False)
    print(f"Wrote {OUT_PORT} ({len(port_df)} rows)")
    print(
        f"  σ current={summary['sigma_current']*100:.2f}%  "
        f"VT_renorm={summary['sigma_VT_renorm']*100:.2f}%  "
        f"RP_inv={summary['sigma_RP_inv_vol']*100:.2f}%  "
        f"RP_ERC={summary['sigma_RP_ERC']*100:.2f}%"
    )
    smci = port_df.loc[port_df["ticker"] == "SMCI"]
    if len(smci):
        r = smci.iloc[0]
        print(
            f"  SMCI: σ={r['sigma']*100:.1f}%  current={r['w_current']*100:.2f}%  "
            f"VT_cap={r['w_VT_capped']*100:.2f}%  RP_inv={r['w_RP_inv_vol']*100:.2f}%  "
            f"RP_ERC={r['w_RP_ERC']*100:.2f}%"
        )

    if not portfolio_only:
        gdf = build_growth_table(
            prices, stocks, window_vol, window_cov, vt_target, smci_cap, growth_cap
        )
        if len(gdf):
            gdf.to_csv(OUT_GROWTH, index=False)
            print(f"Wrote {OUT_GROWTH} ({len(gdf)} rows)")
            print(gdf[["ticker", "sigma", "w_VT_capped", "w_RP_inv_vol", "w_RP_ERC"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Vol targeting vs risk parity analytics")
    ap.add_argument("--window-vol", type=int, default=21)
    ap.add_argument("--window-cov", type=int, default=63)
    ap.add_argument("--target-vol", type=float, default=DEFAULT_VT_TARGET)
    ap.add_argument("--smci-cap", type=float, default=DEFAULT_SMCI_CAP)
    ap.add_argument("--other-cap", type=float, default=DEFAULT_OTHER_CAP)
    ap.add_argument("--growth-cap", type=float, default=DEFAULT_GROWTH_CAP)
    ap.add_argument("--portfolio-only", action="store_true")
    args = ap.parse_args()
    run(
        window_vol=args.window_vol,
        window_cov=args.window_cov,
        vt_target=args.target_vol,
        smci_cap=args.smci_cap,
        other_cap=args.other_cap,
        growth_cap=args.growth_cap,
        portfolio_only=args.portfolio_only,
    )


if __name__ == "__main__":
    main()

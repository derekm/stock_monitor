#!/usr/bin/env python3
"""
rolling_window_analysis.py — Rolling vol, beta, Sharpe, max-DD, dual-screen stability.

Usage:
  python rolling_window_analysis.py
  python rolling_window_analysis.py --window 63 --universe growth
  python rolling_window_analysis.py --save
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
STOCKS = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
OUT = DATA_DIR / "rolling_window_metrics.parquet"
OUT_STAB = DATA_DIR / "rolling_screen_stability.parquet"


def resolve(universe: str) -> list[str]:
    stocks = pd.read_parquet(STOCKS)
    if universe == "portfolio" and HOLDINGS.exists():
        return pd.read_parquet(HOLDINGS)["ticker"].tolist()
    if universe in ("growth", "growth_tech"):
        return stocks.loc[stocks.get("growth_tech_index", False) == True, "ticker"].tolist()
    if universe == "defensive":
        return stocks.loc[stocks.get("defensive_value_index", False) == True, "ticker"].tolist()
    if universe == "aerospace":
        mask = stocks["sector"].isin(["Industrials", "Information Technology"]) & (
            stocks["industry"].astype(str).str.contains("Aerospace|Defense|Semiconductor|Electronic", case=False, na=False)
            | stocks.get("growth_sleeve", pd.Series(dtype=object)).isin(["launch_services", "starlink_supply", "maritime_launch"])
        )
        return stocks.loc[mask, "ticker"].tolist()
    return stocks["ticker"].tolist()


def rolling_metrics(rets: pd.Series, window: int) -> pd.DataFrame:
    r = rets.dropna()
    out = pd.DataFrame(index=r.index)
    out["vol"] = r.rolling(window).std() * np.sqrt(252)
    out["ret"] = r.rolling(window).mean() * 252
    out["sharpe"] = out["ret"] / out["vol"].replace(0, np.nan)
    # rolling max drawdown on price path
    px = np.exp(r.cumsum())
    roll_max = px.rolling(window, min_periods=1).max()
    out["max_dd"] = (px / roll_max - 1).rolling(window).min()
    return out


def run(universe: str = "portfolio", window: int = 63, save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices = prices.rename(columns={"adj_close": "close"})
    prices["date"] = pd.to_datetime(prices["date"])
    tickers = resolve(universe)
    wide = (prices[prices.ticker.isin(tickers)]
            .pivot_table(index="date", columns="ticker", values="close")
            .sort_index().ffill())
    rets = np.log(wide / wide.shift(1))

    # market proxy: equal-weight of available
    mkt = rets.mean(axis=1)

    rows = []
    for t in rets.columns:
        rm = rolling_metrics(rets[t], window)
        # beta vs equal-weight
        cov = rets[t].rolling(window).cov(mkt)
        var = mkt.rolling(window).var()
        beta = cov / var.replace(0, np.nan)
        last = rm.dropna().iloc[-1] if len(rm.dropna()) else None
        if last is None:
            continue
        rows.append({
            "universe": universe,
            "ticker": t,
            "window": window,
            "vol": float(last["vol"]),
            "ann_ret": float(last["ret"]),
            "sharpe": float(last["sharpe"]) if pd.notna(last["sharpe"]) else np.nan,
            "max_dd": float(last["max_dd"]),
            "beta": float(beta.dropna().iloc[-1]) if beta.dropna().size else np.nan,
            "vol_stability": float(rm["vol"].dropna().std()) if rm["vol"].dropna().size > 5 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("vol")
    print(f"=== Rolling {window}d metrics · {universe} ({len(df)} names) ===")
    print(df.head(15).to_string(index=False))
    if save:
        df.to_parquet(OUT)
        print(f"Wrote {OUT}")

    # rolling dual-screen stability from history if present
    hist = DATA_DIR / "preferred_metrics_history.parquet"
    if hist.exists():
        h = pd.read_parquet(hist)
        h["as_of_date"] = pd.to_datetime(h["as_of_date"])
        # for each ticker, fraction of dates passing each screen
        g = h.groupby("ticker").agg(
            n_dates=("as_of_date", "count"),
            buffett_rate=("buffett_pass", "mean"),
            trifecta_rate=("trifecta_pass", "mean"),
            dual_rate=("decision", lambda s: (s == "INCLUDE_CORE").mean()),
            median_composite=("composite_score", "median"),
            composite_std=("composite_score", "std"),
        ).reset_index()
        g = g.sort_values("median_composite", ascending=False)
        g.to_parquet(OUT_STAB)
        print("\n=== Screen stability (through fundamentals history) ===")
        print(g.head(12).to_string(index=False))
        print(f"Wrote {OUT_STAB}")
    return df


def main():
    ap = argparse.ArgumentParser()
    add_index_args(ap, default="portfolio")
    ap.add_argument("--window", type=int, default=63)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run((','.join(resolve_index_names_from_args(args, default_index='portfolio')) or 'portfolio'), args.window, save=True)


if __name__ == "__main__":
    main()

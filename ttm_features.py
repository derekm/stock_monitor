#!/usr/bin/env python3
"""
ttm_features.py — Build TTM-ready multivariate panels from daily_prices.parquet.

Granite TTM works best with:
  - Consistent business-day frequency
  - Per-channel scaling (handled at forecast time)
  - Multiple channels: price, volume, returns, volatility, simple indicators
  - Optional cross-sectional peers / index as extra channels

Usage:
  from ttm_features import build_panel, build_multivariate_bundle
  python ttm_features.py --ticker MOS --save
  python ttm_features.py --index portfolio --save
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
PANEL_DIR = DATA_DIR / "ttm_panels"


def load_ohlcv(tickers: Optional[list[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(PRICES_FILE)
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    return df.sort_values(["ticker", "date"])


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ma_up = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def enrich_ticker(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add channels useful for multivariate TTM on a single ticker."""
    df = ohlcv.sort_values("date").copy()
    c = df["close"]
    df["ret_1"] = np.log(c / c.shift(1))
    df["ret_5"] = np.log(c / c.shift(5))
    df["ma_10"] = c.rolling(10, min_periods=5).mean()
    df["ma_20"] = c.rolling(20, min_periods=10).mean()
    df["ma_ratio"] = c / df["ma_20"]
    df["vol_20"] = df["ret_1"].rolling(20, min_periods=10).std() * np.sqrt(252)
    df["rsi_14"] = _rsi(c, 14)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["hl_range"] = hl / c
    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(20, min_periods=5).mean())
        / df["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    )
    # Typical price & VWAP-ish
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def build_panel(
    ticker: str,
    prices: Optional[pd.DataFrame] = None,
    start: Optional[pd.Timestamp] = None,
    channels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Univariate+indicator panel for one ticker.
    Index = date, columns = channels. Forward-fills small gaps on business days.
    """
    if prices is None:
        prices = load_ohlcv([ticker])
    sub = prices[prices["ticker"] == ticker].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = enrich_ticker(sub)
    if start is not None:
        s = start.date() if hasattr(start, "date") else start
        sub = sub[sub["date"] >= s]
    sub = sub.set_index("date").sort_index()
    # Reindex to business days for consistent frequency (kept as datetime.date)
    if len(sub) >= 2:
        bidx = [d.date() for d in pd.bdate_range(sub.index.min(), sub.index.max())]
        sub = sub.reindex(bidx)
        # price columns interpolate lightly; indicators recompute would be ideal but ffill is OK for gaps
        for col in ["open", "high", "low", "close", "typical", "ma_10", "ma_20"]:
            if col in sub.columns:
                sub[col] = sub[col].ffill()
        sub["volume"] = sub["volume"].fillna(0)
    default_channels = [
        "close", "volume", "ret_1", "vol_20", "rsi_14", "ma_ratio", "hl_range", "volume_z",
    ]
    ch = channels or default_channels
    ch = [c for c in ch if c in sub.columns]
    panel = sub[ch].dropna(how="all")
    return panel


def build_multivariate_bundle(
    tickers: list[str],
    prices: Optional[pd.DataFrame] = None,
    mode: str = "close_only",
    start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Multivariate panel across tickers.
    mode:
      close_only  — columns = ticker closes (classic cross-section)
      full        — stacked channels as ticker:channel
    """
    if prices is None:
        prices = load_ohlcv(tickers)
    if mode == "close_only":
        wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
        wide = wide[[t for t in tickers if t in wide.columns]]
        if start is not None:
            s = start.date() if hasattr(start, "date") else start
            wide = wide[wide.index >= s]
        if len(wide) >= 2:
            bidx = [d.date() for d in pd.bdate_range(wide.index.min(), wide.index.max())]
            wide = wide.reindex(bidx).ffill()
        return wide.dropna(how="all")

    # full channel mode
    parts = []
    for t in tickers:
        p = build_panel(t, prices=prices, start=start)
        if p.empty:
            continue
        p = p.add_prefix(f"{t}:")
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    bundle = pd.concat(parts, axis=1).sort_index()
    return bundle.dropna(how="all")


def scale_panel(panel: pd.DataFrame, method: str = "zscore") -> tuple[pd.DataFrame, dict]:
    """Per-channel scaling; returns scaled panel + params for inverse transform."""
    params = {}
    out = panel.copy()
    for col in out.columns:
        s = out[col]
        if method == "zscore":
            mu, sd = s.mean(), s.std()
            if sd is None or sd == 0 or np.isnan(sd):
                sd = 1.0
            params[col] = ("zscore", float(mu), float(sd))
            out[col] = (s - mu) / sd
        elif method == "minmax":
            lo, hi = s.min(), s.max()
            span = hi - lo if hi != lo else 1.0
            params[col] = ("minmax", float(lo), float(span))
            out[col] = (s - lo) / span
        else:  # none
            params[col] = ("none", 0.0, 1.0)
    return out, params


def inverse_scale(values: np.ndarray, col: str, params: dict) -> np.ndarray:
    kind, a, b = params[col]
    if kind == "zscore":
        return values * b + a
    if kind == "minmax":
        return values * b + a
    return values


def save_panel(panel: pd.DataFrame, name: str) -> Path:
    PANEL_DIR.mkdir(exist_ok=True)
    path = PANEL_DIR / f"{name}.parquet"
    panel.to_parquet(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build TTM feature panels")
    add_ticker_args(parser)
    add_index_args(parser, default="fertilizer")
    parser.add_argument("--mode", choices=["close_only", "full"], default="close_only")
    add_save_arg(parser)
    args = parser.parse_args()

    tickers = resolve_tickers_from_args(args, default_index="fertilizer")
    print(f"Building panel for {tickers} mode={args.mode}")
    if len(tickers) == 1 and args.mode == "full":
        panel = build_panel(tickers[0])
    else:
        panel = build_multivariate_bundle(tickers, mode=args.mode)
    print(f"  shape={panel.shape}  {panel.index.min().date() if len(panel) else 'n/a'} → "
          f"{panel.index.max().date() if len(panel) else 'n/a'}")
    print(f"  columns={list(panel.columns)[:12]}{'...' if panel.shape[1] > 12 else ''}")
    if args.save and len(panel):
        name = args.index or "_".join(tickers[:4])
        path = save_panel(panel, name)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()

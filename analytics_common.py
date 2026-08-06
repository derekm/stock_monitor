#!/usr/bin/env python3
"""analytics_common.py — shared Polars/pandas loaders and return helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

DATA_DIR = Path(__file__).resolve().parent


def prices_path(prefer_clean: bool = True) -> Path:
    clean = DATA_DIR / "daily_prices_clean.parquet"
    raw = DATA_DIR / "daily_prices.parquet"
    if prefer_clean and clean.exists():
        return clean
    return raw


def load_prices_pandas(prefer_clean: bool = True, tickers: Optional[list[str]] = None) -> pd.DataFrame:
    path = prices_path(prefer_clean)
    if HAS_POLARS:
        lf = pl.scan_parquet(str(path)).with_columns(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
        )
        if tickers:
            lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
        df = lf.collect().to_pandas()
    else:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if tickers:
            df = df[df["ticker"].isin([t.upper() for t in tickers])]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def load_adj_prices_pandas(tickers: Optional[list[str]] = None) -> pd.DataFrame:
    """Split/dividend-adjusted closes from daily_prices.parquet (adj_close).

    Return math for long-horizon backtests must use adj_close — raw ``close``
    carries split artifacts (a 4:1 split looks like a -75% day). Prefer this
    over load_prices_pandas for any engine that computes returns.
    """
    path = DATA_DIR / "daily_prices.parquet"
    if HAS_POLARS:
        lf = pl.scan_parquet(str(path)).select(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("ticker"),
            pl.col("adj_close").cast(pl.Float64, strict=False),
        )
        if tickers:
            lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
        df = lf.collect().to_pandas()
    else:
        df = pd.read_parquet(path, columns=cols)
        if tickers:
            df = df[df["ticker"].isin([t.upper() for t in tickers])]
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"adj_close": "close"})
    return df.sort_values(["ticker", "date"])


def wide_closes(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def to_date_keys(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Cast daily date-key columns to python ``datetime.date`` so the parquet
    writer serializes them as DATE (no time component), not TIMESTAMP.

    Daily date-keys (trade dates, as-of dates, snapshot dates) should be DATE,
    not midnight-stamped TIMESTAMP. Intraday event times (fills, alerts,
    last_updated) must NOT be passed here.
    """
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def simple_returns(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.pct_change()


def clip_returns(rets: pd.DataFrame, clip: float = 0.35) -> pd.DataFrame:
    if clip and clip > 0:
        return rets.clip(lower=-clip, upper=clip)
    return rets


def load_membership() -> pd.DataFrame:
    p = DATA_DIR / "monitored_stocks.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_preferred() -> pd.DataFrame:
    p = DATA_DIR / "preferred_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def ann_stats(rets: pd.Series, rf: float = 0.04) -> dict:
    r = rets.dropna()
    if len(r) < 5:
        return {}
    ann_ret = float((1 + r.mean()) ** 252 - 1) if abs(r.mean()) < 0.5 else float(r.mean() * 252)
    # use compound for total path if levels available — mean*252 is ok for clipped daily
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "n": int(len(r))}

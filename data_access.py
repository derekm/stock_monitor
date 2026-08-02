#!/usr/bin/env python3
"""
data_access.py — Shared loaders for parquet/CSV tables used across programs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
HOLDINGS_FILE = DATA_DIR / "portfolio_holdings.parquet"
TRADES_FILE = DATA_DIR / "trades.parquet"
TRADES_FILE_ALT = DATA_DIR.parent / "trades.parquet"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.parquet"
SECTOR_PRICES_FILE = DATA_DIR / "sector_prices.parquet"


def load_stocks() -> pd.DataFrame:
    if not STOCKS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(STOCKS_FILE)


def load_prices(
    tickers: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not PRICES_FILE.exists():
        return pd.DataFrame(columns=["date", "ticker", "close"])
    cols = list(columns) if columns else None
    df = pd.read_parquet(PRICES_FILE, columns=cols)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if tickers is not None:
        df = df[df["ticker"].isin([str(t).upper() for t in tickers])]
    return df


def load_holdings() -> pd.DataFrame:
    if not HOLDINGS_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(HOLDINGS_FILE)


def load_trades() -> pd.DataFrame:
    for path in (TRADES_FILE, TRADES_FILE_ALT):
        if path.exists():
            df = pd.read_parquet(path)
            if "filled_datetime" in df.columns:
                df["filled_datetime"] = pd.to_datetime(df["filled_datetime"])
            return df
    return pd.DataFrame()


def load_fundamentals(latest: bool = True) -> pd.DataFrame:
    if not FUNDAMENTALS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FUNDAMENTALS_FILE)
    if latest and "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        df = df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
    return df


def price_matrix(
    tickers: Sequence[str] | None = None,
    field: str = "close",
) -> pd.DataFrame:
    """Wide date x ticker matrix of a price field."""
    df = load_prices(tickers=tickers, columns=["date", "ticker", field])
    if df.empty:
        return pd.DataFrame()
    return (
        df.pivot_table(index="date", columns="ticker", values=field)
        .sort_index()
        .ffill()
    )

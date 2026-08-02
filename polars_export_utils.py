"""polars_export_utils.py — helpers for large parquet → records without pandas copies."""
from __future__ import annotations
from pathlib import Path
import polars as pl

def tail_prices_records(path: Path, tickers: list[str] | None, days: int = 420, limit: int = 80000) -> list[dict]:
    lf = pl.scan_parquet(str(path)).with_columns(pl.col("date").cast(pl.Date, strict=False))
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
    mx = lf.select(pl.col("date").max()).collect().item()
    if mx is not None:
        # polars duration
        lf = lf.filter(pl.col("date") >= pl.lit(mx) - pl.duration(days=days))
    df = lf.select(["date", "ticker", "close", "volume"]).collect()
    if df.height > limit:
        df = df.tail(limit)
    df = df.with_columns(pl.col("date").cast(pl.Utf8))
    return df.to_dicts()

#!/usr/bin/env python3
"""Partition daily_prices.parquet by year/month for faster DuckDB-Wasm queries.

Creates a partitioned layout:
daily_prices_partitioned/
  year=1962/month=1/data.parquet
  year=1962/month=2/data.parquet
  ...
  year=2026/month=8/data.parquet
"""

import polars as pl
from pathlib import Path

PARTITION_DIR = Path(__file__).parent / "daily_prices_partitioned"

def partition_daily_prices():
    print("Reading daily_prices.parquet...")
    df = pl.read_parquet("daily_prices.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Ensure date is Date type
    df = df.with_columns(pl.col("date").cast(pl.Date))
    
    # Add partition columns
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])
    
    print(f"Writing partitioned data to {PARTITION_DIR}...")
    PARTITION_DIR.mkdir(exist_ok=True)
    
    # Write partitioned by year/month
    df.write_parquet(
        PARTITION_DIR,
        partition_by=["year", "month"],
        use_pyarrow=True,
        pyarrow_options={"compression": "zstd", "compression_level": 3}
    )
    
    # Verify
    parts = list(PARTITION_DIR.rglob("*.parquet"))
    print(f"Created {len(parts)} partition files")
    
    # Quick stats
    total_rows = 0
    for p in parts[:5]:  # sample first 5
        df_part = pl.read_parquet(p)
        total_rows += len(df_part)
    print(f"Sample partitions rows: {total_rows}")
    print("Done!")

if __name__ == "__main__":
    partition_daily_prices()
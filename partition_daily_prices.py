#!/usr/bin/env python3
"""Partition daily_prices/ by year/month for faster DuckDB-Wasm queries.

Creates a partitioned layout:
daily_prices_partitioned/
  year=1962/month=1/data.parquet
  year=1962/month=2/data.parquet
  ...
  year=2026/month=8/data.parquet
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

PARTITION_DIR = Path(__file__).parent / "daily_prices_partitioned"

def partition_daily_prices():
    print("Reading daily_prices/...")
    df = pd.read_parquet("daily_prices/")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Add partition columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    print(f"Writing partitioned data to {PARTITION_DIR}...")

    # Use pyarrow's write_to_dataset for proper hive partitioning
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=str(PARTITION_DIR),
        partition_cols=["year", "month"],
        compression="zstd",
        compression_level=3,
        existing_data_behavior="overwrite_or_ignore"
    )

    # Verify
    parts = list(PARTITION_DIR.rglob("*.parquet"))
    print(f"Created {len(parts)} partition files")

    # Quick stats
    total_rows = 0
    for p in parts[:5]:  # sample first 5
        df_part = pd.read_parquet(p)
        total_rows += len(df_part)
    print(f"Sample partitions rows: {total_rows}")
    print("Done!")

if __name__ == "__main__":
    partition_daily_prices()
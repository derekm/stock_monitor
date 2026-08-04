# data_integrity_deep.py

data_integrity_deep.py — deeper price/fundamental integrity.

## Why it exists (rationale)

Deeper integrity pass (Polars-first) for large tables; complements `data_integrity.py` on fundamentals history and preferred-metrics consistency.

## Usage

```bash
python data_integrity_deep.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
  - `daily_prices_clean.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
  - `monitored_stocks.parquet`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`


## Related programs

- [docs/data_integrity.md](data_integrity.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

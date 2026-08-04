# data_integrity.py

data_integrity.py — Price/fundamental integrity utilities (Polars-first).

## Why it exists (rationale)

Validates and repairs price/fundamental tables (jump audits, clean copies) — guards against the independent-vs-synthetic correlation failure mode and bad corporate-action handling.

## Usage

```bash
python data_integrity.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
  - `daily_prices_clean.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
  - `fundamentals_history.parquet`
  - `fundamentals_pit.parquet`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `price_jump_audit.csv`


## Related programs

- [docs/data_integrity_deep.md](data_integrity_deep.md)
- [docs/update_prices.md](update_prices.md)
- [docs/backfill_historical.md](backfill_historical.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

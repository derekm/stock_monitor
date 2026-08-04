# backfill_constituents.py

backfill_constituents.py — fill REAL multi-snapshot fundamentals + price history for S&P 500 constituents that are missing from our store, using yfinance.

## Why it exists (rationale)

Fills multi-snapshot price + fundamentals history for S&P 500 constituents absent from our store, so S&P-tracking analytics (`sp_universe_tracking`, `sp_index_methodology`) have real data instead of seeds.

## Usage

```bash
python backfill_constituents.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
  - `daily_prices_yfinance.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
  - `fundamentals_yfinance.parquet`
  - `sp500_constituents.parquet`


## Related programs

- [docs/parse_sp500.md](parse_sp500.md)
- [docs/sp_universe_tracking.md](sp_universe_tracking.md)
- [docs/update_fundamentals.md](update_fundamentals.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

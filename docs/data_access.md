# data_access.py

data_access.py — Shared loaders for parquet/CSV tables used across programs.

## Why it exists (rationale)

Central parquet/CSV loaders used by nearly every program — the single place that knows the on-disk layout of `daily_prices`, `fundamentals`, `monitored_stocks`, `portfolio_holdings`, `sector_prices`, `trades`.

## Usage

```bash
python data_access.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
  - `sector_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
  - `monitored_stocks.parquet`
  - `portfolio_holdings.parquet`
  - `trades.parquet`


## Related programs

- [docs/cli_common.md](cli_common.md)
- all stock_monitor programs (via `data_access`/`cli_common`)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

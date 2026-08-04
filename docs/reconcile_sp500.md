# reconcile_sp500.py

Reconcile sp500_member / sp500_sector / sp500_date_added in monitored_stocks.parquet against the authoritative sp500_constituents.parquet.

## Why it exists (rationale)

Reconciles `sp500_member`/`sp500_sector`/`sp500_date_added` in `monitored_stocks.parquet` against the authoritative `sp500_constituents.parquet`.

## Usage

```bash
python reconcile_sp500.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/parse_sp500.md](parse_sp500.md)
- [docs/sp_universe_tracking.md](sp_universe_tracking.md)
- [docs/manage_stocks.md](manage_stocks.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

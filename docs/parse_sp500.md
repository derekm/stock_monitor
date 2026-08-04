# parse_sp500.py

Parse the S&P 500 constituents table from the downloaded Wikipedia HTML.

## Why it exists (rationale)

Parses the S&P 500 constituents table from Wikipedia HTML into `sp500_constituents.parquet` — upstream of all S&P-tracking analytics.

## Usage

```bash
python parse_sp500.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/parse_sp500_changes.md](parse_sp500_changes.md)
- [docs/sp_universe_tracking.md](sp_universe_tracking.md)
- [docs/reconcile_sp500.md](reconcile_sp500.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

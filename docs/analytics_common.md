# analytics_common.py

analytics_common.py — shared Polars/pandas loaders and return helpers.

## Why it exists (rationale)

Shared Polars/pandas loaders and return helpers used by analytics programs.

## Usage

```bash
python analytics_common.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/data_access.md](data_access.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

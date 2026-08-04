# polars_export_utils.py

polars_export_utils.py — helpers for large parquet → records without pandas copies.

## Why it exists (rationale)

Helpers to stream large parquet → JSON records without pandas copies; used by `export_dashboard_data` / `build_data_catalog`.

## Usage

```bash
python polars_export_utils.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/export_dashboard_data.md](export_dashboard_data.md)
- [docs/build_data_catalog.md](build_data_catalog.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

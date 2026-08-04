# export_dashboard_data.py

Export key analytics tables to dashboard_data/data.json for DuckDB-Wasm / static UI.

## Why it exists (rationale)

Exports key analytics tables into `dashboard_data/data.json` for the static/DuckDB-Wasm UI; the last step of `start_dashboard.sh`.

## Usage

```bash
python export_dashboard_data.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/build_data_catalog.md](build_data_catalog.md)
- [docs/pipeline_service.md](pipeline_service.md)
- `start_dashboard.sh`
- [docs/granite_service.md](granite_service.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

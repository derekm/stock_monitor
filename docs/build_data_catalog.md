# build_data_catalog.py

Build data_catalog.json listing all CSV/Parquet resources for the dashboard.

## Why it exists (rationale)

Builds `data_catalog.json` listing all CSV/Parquet resources for the dashboard's DuckDB-Wasm catalog; run by `start_dashboard.sh`.

## Usage

```bash
python build_data_catalog.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/export_dashboard_data.md](export_dashboard_data.md)
- [docs/pipeline_service.md](pipeline_service.md)
- `start_dashboard.sh`
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

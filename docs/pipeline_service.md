# pipeline_service.py

pipeline_service.py — HTTP control plane to rerun data jobs for the dashboard.

## Why it exists (rationale)

HTTP control plane that reruns data jobs (prices, backfill, analytics, export, forecast_bt, monte_carlo, all) as subprocesses — the dashboard's job runner.

## Usage

```bash
python pipeline_service.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/analytics_service.md](analytics_service.md)
- [docs/granite_service.md](granite_service.md)
- [docs/export_dashboard_data.md](export_dashboard_data.md)
- `start_dashboard.sh`
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

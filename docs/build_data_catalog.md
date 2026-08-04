# build_data_catalog.py

Builds `dashboard_data/data_catalog.json`, which lists every CSV/Parquet
resource in the repo so the dashboard's DuckDB-Wasm catalog can discover tables.

## Why it exists (rationale)

The static dashboard loads tables via DuckDB-Wasm from a JSON catalog rather
than scanning the filesystem at runtime. This script walks `DATA_DIR`, skips
irrelevant dirs (`.git`, `node_modules`, `logs`, `__pycache__`, `dashboard_data`),
and emits the catalog that `export_dashboard_data.py` and the dashboard rely on.
Run automatically by `start_dashboard.sh`.

## Usage

```bash
python build_data_catalog.py
```

Flags: none. Writes `dashboard_data/data_catalog.json`.

## Outputs

- `dashboard_data/data_catalog.json` — list of available CSV/Parquet resources

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [export_dashboard_data.md](export_dashboard_data.md) — builds `data.json`
- `start_dashboard.sh` — runs both at startup
- [pipeline_service.md](pipeline_service.md)

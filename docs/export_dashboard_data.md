# export_dashboard_data.py

Export the key analytics tables into `dashboard_data/data.json` for the
DuckDB-Wasm / static dashboard, then regenerate the data catalog.

## Why it exists (rationale)

The static dashboard can't run Python; it reads a single JSON payload of all
analytics tables via DuckDB-Wasm. This script gathers the latest CSV/parquet
analytics artifacts (regimes, risk, screens, forecasts, indexes, fundamentals),
serializes them into `data.json`, and finally calls `build_data_catalog.py` so
the catalog reflects what was just exported. It is one of the four services
launched by `start_dashboard.sh`.

## What it bundles

- Regime outputs (`hmm_*`, `kalman_*`, `vix_term_structure*`, `posterior_entropy*`)
- Risk (`risk_metrics`, `tail_risk_hedge*`, `rolling_corr*`, `regime_corr_breakdown`)
- Screens / decisions (`threshold_logic_screen`, `preferred_metrics`, decision notes)
- Indexes / fundamentals (`fertilizer_index`, latest fundamentals, value-trifecta)
- Forecasts / anomalies if present

Each table is capped (`df_records` cap 500 rows) and serialized to records.

## Usage

```bash
python export_dashboard_data.py
```

Flags: none. Writes `dashboard_data/data.json` and runs `build_data_catalog.py`.

## Outputs

- `dashboard_data/data.json` — consolidated dashboard payload
- (indirect) `dashboard_data/data_catalog.json` via `build_data_catalog.py`

## Related programs

- [build_data_catalog.md](build_data_catalog.md) — catalog it triggers
- [analytics_service.md](analytics_service.md) — `POST /run/export-dashboard` calls this
- `start_dashboard.sh` — launches it at boot
- [pipeline_service.md](pipeline_service.md)

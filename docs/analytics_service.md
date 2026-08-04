# analytics_service.py

HTTP microservice that lets the dashboard trigger daily analytics jobs and read
analytics tables. Together with `granite_service.py` and `pipeline_service.py`
it is one of the three services launched by `start_dashboard.sh`.

## Why it exists (rationale)

The static dashboard cannot run Python directly, so it needs a local API to (a)
kick off the heavy daily jobs and (b) fetch the resulting tables as JSON. This
service is the "ops" hub; `pipeline_service.py` is the lower-level job runner it
often delegates to.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness |
| GET | `/tables` | list available parquet/csv artifacts |
| GET | `/table?name=preferred_metrics` | JSON rows of a named table |
| GET | `/dual-pass` | INCLUDE_CORE rows |
| GET | `/rolling?universe=portfolio&window=63` | recompute + return rolling stats |
| GET | `/aerospace` | supply-chain membership |
| POST | `/run/update-prices` | refresh prices |
| POST | `/run/backfill` | historical backfill |
| POST | `/run/fundamentals-snapshot` | history scores |
| POST | `/run/preferred-metrics` | refresh scores |
| POST | `/run/rolling` | rolling windows |
| POST | `/run/growth-analytics` | growth-tech suite |
| POST | `/run/alerts` | evaluate alerts |
| POST | `/run/export-dashboard` | rewrite `dashboard_data/data.json` |
| POST | `/run/all-daily` | price path + metrics + export |

## Usage

```bash
python analytics_service.py --port 8765
```

> The script default port is **8765**. `start_dashboard.sh` overrides it to
> **8767** so the static dashboard keeps 8765. The dashboard Ops tab posts to
> these endpoints.

## Outputs

None written directly. It triggers other programs (which write the analytics
CSVs) and reads them back. See [SCHEMAS.md](SCHEMAS.md).

## Related programs

- [granite_service.md](granite_service.md) — live forecasts API
- [pipeline_service.md](pipeline_service.md) — low-level job runner
- [export_dashboard_data.md](export_dashboard_data.md) — builds `data.json`
- `start_dashboard.sh` — launches all four services

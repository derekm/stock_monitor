# granite_service.py

Live Granite / fallback forecast microservice for the dashboard (port **5055**).

## Why it exists (rationale)

The static dashboard needs forecasts on demand. Unlike `export_dashboard_data`
(which ships a static `data.json`), this service **always computes** forecasts at
request time (never serves a stale parquet). It supports multivariate channels,
index peers, correlated/uncorrelated peer sets, days-ago history windows, and
multi-index membership via `index_registry`. One of the three services launched
by `start_dashboard.sh`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness |
| GET | `/forecast?ticker=AEP&horizon=20&...` | compute a fresh forecast |
| GET | `/forecast_multi` | multi-ticker batch |
| GET | `/index?name=portfolio` | index-level forecast |
| GET | `/coverage` | ticker coverage |

## Usage

```bash
python granite_service.py --port 5055
```

> Default port **5055**. `start_dashboard.sh` launches it there.

## Outputs

None written to disk (computes in-memory, returns JSON). Consumes the latest
Granite checkpoint under `checkpoints/`.

## Related programs

- [forecast_granite.md](forecast_granite.md) — the forecast kernel it calls
- [granite_daily.md](granite_daily.md) / [ttm_backfill.md](ttm_backfill.md) — checkpoint source
- [analytics_service.md](analytics_service.md) / [pipeline_service.md](pipeline_service.md)
- `start_dashboard.sh`

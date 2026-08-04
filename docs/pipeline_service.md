# pipeline_service.py

HTTP control plane to re-run data jobs for the dashboard (port **5056**).

## Why it exists (rationale)

The static dashboard needs to trigger heavy jobs (price refresh, backfill,
analytics, export, forecast backtests, Monte Carlo) without a Python runtime.
This service accepts job requests and spawns the corresponding existing program
as a subprocess, streaming logs to `logs/pipeline_*.log`. It is the low-level job
runner; `analytics_service.py` often delegates to it. One of the four services
launched by `start_dashboard.sh`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/run` | `{"job": "prices"|"backfill"|"analytics"|"export"|"forecast_bt"|"monte_carlo"|"all"|...}` |
| GET | `/jobs` | list known jobs |
| GET | `/status` | running/last job state |
| GET | `/health` | liveness |

## Usage

```bash
python pipeline_service.py --port 5056
```

> Default port **5056**. `start_dashboard.sh` launches it there. Logs under
> `logs/pipeline_*.log`.

## Outputs

None written directly (subprocesses write their own artifacts). See each job's
program doc.

## Related programs

- [analytics_service.md](analytics_service.md) — higher-level ops hub
- [granite_service.md](granite_service.md) — forecasts API
- [export_dashboard_data.md](export_dashboard_data.md) — the `export` job
- `start_dashboard.sh`

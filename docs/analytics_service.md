# analytics_service.py

HTTP microservice so the dashboard can trigger daily jobs.

**Standard launch:** run `./start_dashboard.sh` — it starts this service (port 8767) along with `granite_service`, `pipeline_service`, and the static dashboard.

Manual (only if launching in isolation):

```bash
python analytics_service.py --port 8765
```

> `start_dashboard.sh` launches it on port **8767** (override); 8765 is the script default.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness |
| GET | `/tables` | list artifacts |
| GET | `/table?name=preferred_metrics` | JSON rows |
| GET | `/dual-pass` | INCLUDE_CORE rows |
| GET | `/rolling?universe=growth` | recompute + return |
| GET | `/aerospace` | supply-chain membership |
| POST | `/run/preferred-metrics` | refresh scores |
| POST | `/run/rolling` | rolling windows |
| POST | `/run/fundamentals-snapshot` | history scores |
| POST | `/run/export-dashboard` | rewrite `dashboard_data/data.json` |
| POST | `/run/all-daily` | preferred + rolling + snapshot + export |

Dashboard **Ops** tab posts to these endpoints.

## Related programs

- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/export_dashboard_data.md](export_dashboard_data.md)
- [docs/pipeline_service.md](pipeline_service.md)
- [docs/granite_service.md](granite_service.md)
- `start_dashboard.sh`
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

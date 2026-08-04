# granite_service.py

HTTP microservice for **Granite TTM / statistical fallback** forecasts consumed by the dashboard Chart.js views.

## Run

**Standard launch:** run `./start_dashboard.sh` — it starts this service (port 5055) along with `pipeline_service`, `analytics_service`, and the static dashboard.

Manual (only if launching in isolation):

```bash
python granite_service.py --host 127.0.0.1 --port 5055
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/tickers` | Monitored tickers |
| GET | `/forecast?tickers=MOS,CF&horizon=10` | Forecast selected names |
| GET | `/forecast/portfolio?horizon=10&from_first_trade=1` | Portfolio (first-trade aware) |
| GET | `/forecast/index?name=fertilizer&horizon=10` | Fertilizer or defensive index |
| GET | `/forecast/sectors?horizon=5` | Sector EW (`SECT_*`) indexes |
| POST | `/forecast` | JSON body `{"tickers":["MOS"],"horizon":10}` |

Response includes:

- `forecasts` — flat rows (horizon, pct_change, …)
- `charts` — per ticker `{ history: [{date, close}], forecast: [{date, close, horizon, pct_change}] }` for line charts

CORS: `Access-Control-Allow-Origin: *` for local dashboard use.

## Dashboard

**Forecasts** tab → set API base URL → **Fetch forecasts**. Plots:

1. History + dashed forecast path (Chart.js line)
2. Horizon % change bar chart

## Notes

- Uses `forecast_granite.load_granite_model` / `forecast_ttm_univariate` (real TTM when installed; else fallback).
- Reads `daily_prices.parquet` (and sector prices when needed).
- Stdlib only for the HTTP layer (`ThreadingHTTPServer`).

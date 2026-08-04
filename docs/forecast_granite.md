# forecast_granite.py

Stock forecasting with IBM Granite Time Series (TTM) — the live forecast engine
of the Granite subsystem.

## Why it exists (rationale)

Produces the rolling price forecasts that feed the dashboard's forecast tab and
the anomaly/decision layer. It builds business-day OHLCV panels with multivariate
channels (price, volume, returns, vol, RSI, peers), loads the latest global
Granite TTM checkpoint, and runs zero-shot (and optional iterative rolling)
forecasts with directional-accuracy + MAE/RMSE/MAPE evaluation.

## Subcommands

- `forecast` — generate forecasts for an index/ticker
- `backtest` — evaluate a config over history (writes `forecast_backtest_metrics.csv`)

## Usage

```bash
python forecast_granite.py forecast --index portfolio --horizon 20 --save
python forecast_granite.py backtest --index portfolio --horizons 5,10,20 --windows 40,60
```

Reads `daily_prices.parquet`, `sector_prices.parquet`, `monitored_stocks.parquet`,
`trades.parquet`, `portfolio_holdings.parquet`. Loads the latest checkpoint from
`checkpoints/`.

## Outputs

- `forecasts_granite.parquet` / `forecasts_granite.csv` — per-ticker forecasts
- `forecast_backtest_metrics.csv` — backtest error metrics

(Schema family: forecast_anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [ttm_features.md](ttm_features.md) — builds the panels it consumes
- [granite_backfill.md](granite_backfill.md) — pretrains the checkpoint
- [granite_daily.md](granite_daily.md) / [granite_forecast.md](granite_forecast.md) — daily + forecast wrappers
- [forecast_reliability.md](forecast_reliability.md)
- [granite_service.md](granite_service.md) — serves these forecasts

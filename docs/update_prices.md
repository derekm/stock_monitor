# update_prices.py

Append daily OHLCV rows to `daily_prices.parquet`.

## Purpose
Keep monitored tickers current for indexes, Fisher quantities (volume), TTM panels, and forecasts.

## Usage
```bash
python update_prices.py --fetch --days 5          # yfinance when network available
python update_prices.py --manual TICK open close [--date YYYY-MM-DD]
python update_prices.py --from-csv prices.csv
```

## Notes
- Merges on `(date, ticker)`; last write wins.
- Prefer full OHLCV. **`volume` is quantity (q)** for Laspeyres/Paasche/Fisher indexes.
- After updates: `python fisher_index.py --universe portfolio --save` and/or `python run_fisher_duckdb.py --universe portfolio --save`.
- TTM panels benefit from consistent business-day history (`ttm_features.py`).

## Related programs

- [docs/backfill_historical.md](backfill_historical.md)
- [docs/fisher_index.md](fisher_index.md)
- [docs/run_fisher_duckdb.md](run_fisher_duckdb.md)
- [docs/ttm_features.md](ttm_features.md)
- [docs/granite_daily.md](granite_daily.md)
- [docs/forecast_granite.md](forecast_granite.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

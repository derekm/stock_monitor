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

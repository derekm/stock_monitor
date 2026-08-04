# backfill_historical.py

Populate `daily_prices.parquet` with historical OHLCV data.

## Why it exists (rationale)

Most downstream work (Fisher quantity indexes, TTM/Granite forecasts,
correlations, backtests) needs 1y+ of clean daily history. This script is the
primary way to acquire it from yfinance, with offline fallbacks for pipeline
testing.

## Usage

```bash
python backfill_historical.py --period 1y
python backfill_historical.py --start 2025-01-01 --end 2026-07-28
python backfill_historical.py --tickers CF,MOS,NTR --period 6mo
python backfill_historical.py --synthetic --days 30    # offline pipeline tests only
python backfill_historical.py --from-csv historical.csv
```

Flags (own argparse):

- `--period` — yfinance period token (`1mo,3mo,6mo,1y,2y,5y,ytd,max`)
- `--start` / `--end` — explicit date range (YYYY-MM-DD)
- `--tickers` — comma-separated subset (default: all active+monitored)
- `--status` — which `monitored_stocks` statuses to include (default active,monitored)
- `--synthetic` — generate random-walk data (testing only)
- `--days` — synthetic history length (default 90)
- `--from-csv` — bulk import from a `date,ticker,open,close,...` CSV
- `--overwrite` — replace conflicting rows (default: merge, newest source wins)
- `--seed` — RNG seed for synthetic data

## Outputs

- `daily_prices.parquet` — merged on `(date, ticker)`; newest source wins;
  existing rows are never deleted unless `--overwrite`.

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Notes

- Prefer real yfinance **volume** (quantity) for Fisher index stability.
- Prefer `--period 2y` when network allows, so the 512-day TTM context is full.
- Synthetic backfill is for sandbox continuity, not decisions.
- After backfill: `python ttm_features.py --index portfolio --save` and/or
  `python cross_asset_analysis.py save-sector-prices`.

## Related programs

- [update_prices.md](update_prices.md) — incremental daily appends
- [ttm_features.md](ttm_features.md) — builds TTM panels from the prices
- [cross_asset_analysis.md](cross_asset_analysis.md) — sector EW prices
- [fisher_index.md](fisher_index.md) / [run_fisher_duckdb.md](run_fisher_duckdb.md)

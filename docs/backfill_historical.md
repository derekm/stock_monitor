# backfill_historical.py

Populate `daily_prices.parquet` with historical OHLCV.

## Purpose
Build 1y+ history so Granite TTM context (≤512 days), Fisher chains, correlations, and backtests have enough depth.

## Usage
```bash
python backfill_historical.py --period 1y
python backfill_historical.py --start 2025-01-01 --end 2026-07-28
python backfill_historical.py --tickers CF,MOS,NTR --period 6mo
python backfill_historical.py --synthetic --days 30    # offline pipeline tests only
python backfill_historical.py --from-csv historical.csv
```

## Notes
- Prefer real yfinance volume for Fisher quantity weights.
- Synthetic paths are for sandbox continuity, not decision-making.
- Rebuild sector EW prices after large backfills: `python cross_asset_analysis.py save-sector-prices`.

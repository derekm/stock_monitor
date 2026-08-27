# update_polygon.py

Daily OHLCV ingest from Polygon.io (production-grade price feed),
key-gated.

## Why it exists (rationale)

Closes the "integrate data sources: Polygon (production)" TODO. The repo
ingests via yfinance (prototyping); Polygon is the production alternative
with a free tier.

## Method

- Requires `POLYGON_API_KEY` env var. Without it, prints how to get a key
  and exits 0 (no crash in the automation).
- Pulls daily bars (adjusted=true) for the monitored universe and appends
  into `daily_prices/` (dedup on date+ticker, keep last).
- Free-tier rate limit respected (0.21s sleep ≈ 5 req/s).
- Output columns match the existing price spine: date, ticker, open, high,
  low, close, adj_close, volume, source='polygon', market_cap.

## Usage

```bash
export POLYGON_API_KEY=...
python update_polygon.py --days 5 --save
```

## Outputs

- Appends to `daily_prices/` (base table).

## Related programs

- `update_prices.py` — the yfinance ingestion path this complements
- `daily_prices/` — the shared price spine
- `run_daily_automation.py` — registered as `polygon_prices` job (runs after
  market close, free-tier safe)

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
  into `daily_prices.parquet` (dedup on date+ticker, keep last).
- Free-tier rate limit respected (0.21s sleep ≈ 5 req/s).

## Usage

```bash
export POLYGON_API_KEY=...
python update_polygon.py --days 5 --save
```

## Outputs

- Appends to `daily_prices.parquet` (base table).

## Related programs

- `update_prices.py` — the yfinance ingestion path this complements
- `daily_prices.parquet` — the shared price spine

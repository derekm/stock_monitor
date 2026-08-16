# acquisition_backfill.py
Detect corporate actions and trigger existing backfill/ingest pipelines.

## Why it exists (rationale)

When an acquisition happens, the target ticker may not exist in `daily_prices.parquet` or `fundamentals.parquet`, leaving the acquirer's history incomplete. This script detects M&A activity from SEC EDGAR (companyfacts M&A tags + filings index) and delegates retrieval to the existing canonical pipelines (`backfill_edgar.py`, `update_polygon.py`, yfinance) rather than reimplementing them. It also registers acquisitions in `corporate_actions.parquet` so `lookthrough_engine.py` can apply pro forma combination during the look-through window.

## Usage

```bash
python acquisition_backfill.py --tickers AAPL MSFT              # scan specific tickers for acquisitions
python acquisition_backfill.py --process PANW CYBR 2026-07-31 2026-03-15  # manually register an acquisition
python acquisition_backfill.py                                  # scan full universe (daily_prices)
```

## Outputs

- `daily_prices.parquet` — backfilled price history for target tickers (schema family: base_table)
- `fundamentals.parquet` — backfilled fundamentals for target tickers (schema family: base_table)
- `corporate_actions.parquet` — acquisition records with look-through windows (schema family: other)

## Related programs

- `backfill_edgar.py` — SEC companyfacts → fundamentals (used for target backfill)
- `update_polygon.py` — Polygon bulk price ingest (primary price backfill)
- `lookthrough_engine.py` — pro forma combination during acquisition windows
- `edgar_lib.py` — shared EDGAR utilities used for CIK resolution
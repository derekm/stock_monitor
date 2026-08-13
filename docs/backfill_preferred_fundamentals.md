# backfill_preferred_fundamentals.py

One-shot **additive** expander for preferred-metrics coverage.

Preferred history is a scored view of `fundamentals.parquet`. This script
deepens fundamentals from every real source, then rebuilds the snapshot.

## Order

1. `backfill_edgar.py` — SEC XBRL, decades, US filers
2. `update_fundamentals.py fetch-history` — yfinance quarterly (~2y, ADRs)
3. `update_polygon_financials.py` — Massive/Polygon vX financials if keyed
4. `fundamentals_history.py snapshot` → `preferred_metrics_history.parquet`
5. `preferred_metrics.py --save` → latest `preferred_metrics.parquet`

Merge rule: **never overwrite a populated cell**. New `(ticker, date)` rows
are appended. Source rank: edgar > manual > yfinance_history >
polygon_financials > yfinance > synthetic.

## Usage

```bash
python backfill_preferred_fundamentals.py
python backfill_preferred_fundamentals.py --tickers AAPL,GOLD,BTI
python backfill_preferred_fundamentals.py --no-polygon
```

`add_ticker.py` calls this with `--tickers` for each onboarded name.

Polygon auth: `POLYGON_API_KEY` or gitignored `massive_credentials.json`
`secret_access_key`. ETFs and some ADRs have no statements — expected.

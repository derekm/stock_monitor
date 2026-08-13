# update_fundamentals.py / update_polygon_financials.py / backfill_edgar.py

Maintain point-in-time quality/value fundamentals for the **full universe**.

## Additive rule

Never overwrite a populated cell. New `(ticker, as_of_date)` rows are appended.
Overlapping dates only **fill NaNs**. Source rank:

`edgar > manual > yfinance_history > polygon_financials > yfinance > synthetic`

## Fetch paths

```bash
python update_fundamentals.py fetch-history          # yfinance quarterly (~2y)
python update_polygon_financials.py                 # Massive/Polygon vX financials
python update_polygon_financials.py --missing-only
python backfill_edgar.py                            # SEC XBRL, decades
python fundamentals_history.py snapshot             # rebuild preferred_metrics_history
python preferred_metrics.py --save                  # latest snapshot
```

`fetch-history`, Polygon, and EDGAR all walk `universe_tickers()` (shock_ride ∪
prices ∪ monitored), not just the 147-name monitored book.

Polygon auth: `POLYGON_API_KEY` or gitignored `massive_credentials.json`
`secret_access_key`. Do not commit the key.

ETFs / some ADRs (QQQ, XLE, BTI, BAYRY, …) have no statements — expected.

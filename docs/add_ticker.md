# add_ticker.py — one-command onboarding

Adds ticker(s) to the universe and runs the full backfill + analytics chain.

1. Look up name / sector / industry (yfinance, overridable)
2. Add to `monitored_stocks.parquet` (idempotent)
3. Price history `--period max`
4. Fundamentals via `backfill_preferred_fundamentals.py`
   (EDGAR → yfinance quarterly → Polygon financials, **additive**;
   then rebuild preferred-metrics history + latest snapshot)
5. Momentum metrics
6. Daily market cap
7. Full `run_daily_automation.py` unless `--no-analytics`

```bash
python add_ticker.py QSR
python add_ticker.py QSR CAG PFE
python add_ticker.py QSR --no-analytics --no-fundamentals
```

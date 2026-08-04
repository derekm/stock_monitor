# _noadj_probe.py

**Developer scratch** — not part of the production pipeline.

Data-quality probe for the `adj_close` column in `daily_prices.parquet`. It
reads the prices parquet, groups by ticker, and reports which tickers are
missing `adj_close` or whose `adj_close` is effectively identical to `close`
(mean relative difference < 1e-4) — i.e. names that are not split/dividend
adjusted.

Informs whether the adjusted-close backfill (`backfill_historical.py`,
`train_adjusted_full.py`) actually has adjustment data to work with. Hard-codes
the parquet path. No persistent outputs; prints the no-adjustment set to stdout.

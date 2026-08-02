# fundamentals_history.py

Time-series fundamentals for **backtesting inclusion / selection theses**.

## Design

- `fundamentals.parquet` is **append-only** by `as_of_date` (not latest-only).
- `backfill` creates synthetic prior quarter-ends from the latest snapshot (for pipeline tests).
- `snapshot` scores every dated row → `preferred_metrics_history.parquet`
- `backtest-screens` counts Buffett / trifecta / dual passes through time → `screen_backtest.csv`

```bash
python fundamentals_history.py backfill --quarters 8
python fundamentals_history.py snapshot
python fundamentals_history.py backtest-screens
python fundamentals_history.py show --ticker MOS
```

Replace backfilled rows with real quarterly fundamentals when available; keep the same schema so screens and sizing stay backtestable.

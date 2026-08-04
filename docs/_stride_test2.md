# _stride_test2.py

**Developer scratch** — not part of the production pipeline.

Second stride / cleaning probe. Unlike `_stride_test.py`, it first **dedupes by
ticker+date** and keeps only the recent stationary tail (~2520 trading days ≈
10y) before scoring windows — testing whether trimming old history changes
forecast scores. This is the experiment that motivated the `recent_trading_days`
tail used in `granite_backfill._clean_price_frame`.

Uses `granite_backfill.score_windows`. No persistent outputs; prints to stdout.

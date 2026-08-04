# _stride_test3.py

**Developer scratch** — not part of the production pipeline.

Third stride / cleaning probe. It dedupes by ticker+date, sorts, then keeps only
the last `RECENT = 2520` rows per ticker before scoring — a tighter version of
the tail-trim experiment in `_stride_test2.py`.

Uses `granite_backfill.score_windows`. No persistent outputs; prints to stdout.

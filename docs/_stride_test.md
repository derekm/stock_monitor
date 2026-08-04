# _stride_test.py

**Developer scratch** — not part of the production pipeline.

Stride / window-overlap probe for the backfill. It loads prices + checkpoint,
builds full-history windows, and scores them to compare how stride (window
spacing) affects the number of windows and forecast scores — tuning the window
extraction used by `granite_backfill`.

Uses `granite_backfill.build_full_history_windows` and `score_windows`. No
persistent outputs; prints to stdout.

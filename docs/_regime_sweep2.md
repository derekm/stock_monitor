# _regime_sweep2.py

**Developer scratch** — not part of the production pipeline.

Second regime-conditioned backfill sweep (variant of `_regime_sweep.py`). It
loads prices + checkpoint and re-runs windows with a different cleaning path
(`_clean_price_frame`) to compare regime/cleaning interactions on forecast
quality.

Uses `granite_backfill`. No persistent outputs; prints to stdout.

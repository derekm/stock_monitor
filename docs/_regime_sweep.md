# _regime_sweep.py

**Developer scratch** — not part of the production pipeline.

Regime-conditioned backfill sweep. It loads prices + the latest global
checkpoint, and re-runs backfill windows across different regime assumptions
(context/horizon from `gd`) to see how regime tagging affects forecast quality —
feeds `ttm_backfill.py`'s regime config.

Uses `granite_backfill`. No persistent outputs; prints to stdout.

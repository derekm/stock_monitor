# _cmp_adj_unadj.py

**Developer scratch** — not part of the production pipeline.

Comparison probe for the adjusted-vs-unadjusted close debate in the TTM
backfill. It imports `granite_backfill` and reuses `_clean_price_frame` to
build two price series per ticker — one with `use_adj=True` (adjusted closes)
and one with `use_adj=False` (raw closes) — then builds training windows via
`gd.CONTEXT` / `gd.HORIZON` and compares them.

Purpose: confirm whether using adjusted vs raw closes materially changes the
window tensors fed to the Granite TTM model. No persistent outputs; prints
comparisons to stdout.

# _p3_debug.py

**Developer scratch** — not part of the production pipeline.

Debug probe for Pass 3 (Granite TTM parameter sweep). It loads `granite_backfill`,
builds the clean adjusted-close price frame, extracts AEP's adjusted close, and
prints basic stats (count, min, max, mean) plus model/context diagnostics to
confirm the data and config used by `pass3_sweep.py` are sane.

Uses `_clean_price_frame` and `gd` from `granite_backfill`. No persistent
outputs; prints to stdout.

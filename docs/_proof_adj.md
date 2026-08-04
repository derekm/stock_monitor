# _proof_adj.py

**Developer scratch** — not part of the production pipeline.

Proof-of-concept comparing adjusted vs unadjusted close windows at stride 1. It
builds per-ticker window tensors from `_clean_price_frame` (with `use_adj` on and
off), using `gd.CONTEXT`/`gd.HORIZON`, to demonstrate the difference the
adjusted-close choice makes at the tensor level.

Uses `granite_backfill` and `granite_daily`. No persistent outputs; prints to
stdout.

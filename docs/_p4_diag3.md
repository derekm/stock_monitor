# _p4_diag3.py

**Developer scratch** — not part of the production pipeline.

Diagnostic for Pass 4 window format. It imports `pass4` and `granite_backfill`,
reads prices, builds the adjusted clean frame using `pass4.RECENT` (the same
`recent_trading_days` window `train_aggregate`/`score_windows` use), extracts AEP
close, and reconstructs windows in the exact `(context, target, ticker)` format —
to confirm the window tensor shape matches what scoring expects.

Uses `pass4` and `granite_backfill._clean_price_frame`. No persistent outputs;
prints to stdout.

# _ptsteps_high2.py

**Developer scratch** — not part of the production pipeline.

Per-ticker training-step allocator probe. It groups full-history windows by
ticker, samples 2 easy + 2 volatile names (F, NVR, AEP, FICO — the volatile ones
are the ones that matter for "more steps"), and experiments with allocating more
training steps to volatile tickers — informing `vol_steps.py`.

Uses `granite_backfill`. No persistent outputs; prints to stdout.

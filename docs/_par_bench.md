# _par_bench.py

**Developer scratch** — not part of the production pipeline.

Parallelism benchmark for the backfill. It groups full-history windows by
ticker, picks ~6 tickers with ~200 windows each (the common case), and times
single-process vs multiprocess training to quantify the speedup from running
separate processes (each with its own model copy) — the production parallelism
strategy, since `torch.compile`/inductor does not work on Windows.

Uses `granite_backfill`. No persistent outputs; prints timing to stdout.

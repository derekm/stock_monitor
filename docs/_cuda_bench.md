# _cuda_bench.py

**Developer scratch** — not part of the production pipeline.

Device-throughput benchmark for TinyTimeMixer (TTM) training on this machine.
It builds full-history windows from `daily_prices.parquet` (AEP repeated to 1024
samples so batch=512 fits), constructs a `TinyTimeMixerForPrediction` model, and
times a training step on the available device (CUDA MX550 vs CPU) to measure
whether the GPU gives a speedup for the small-batch backfill workload.

Related to `bench_backfill.py` (which benchmarks the broader rolling-window
backfill). No persistent outputs; prints timing to stdout.

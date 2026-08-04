# _par_test.py

**Developer scratch** — not part of the production pipeline.

Parallelism test for the backfill. It builds full-history windows, loads the
latest global checkpoint, and runs a background `gpu_poll` while training to
capture per-second GPU utilization — measuring whether a parallel/multiprocess
backfill keeps the GPU busy versus stalling.

Companion to `_par2_test.py` / `_par_bench.py`. No persistent outputs; prints to
stdout.

# _par2_test.py

**Developer scratch** — not part of the production pipeline.

Throughput/profiling probe for the parallel backfill. It builds full-history
windows, loads the latest global checkpoint, and runs a background GPU poll
(`gpu_poll`) while training, capturing per-second GPU utilization into a list —
used to measure whether multiprocessing backfill actually saturates the GPU.

Companion to `_cuda_bench.py` / `_feed_test.py`. No persistent outputs; prints
to stdout.

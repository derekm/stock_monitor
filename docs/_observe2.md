# _observe2.py

**Developer scratch** — not part of the production pipeline.

A second live process/GPU observer (variant of `_observe.py`). Hard-codes a
different PID (45880) and polls per-core CPU usage (`psutil`) and GPU utilization
(`nvidia-smi`) on an interval for a second concurrent run.

Hard-codes the PID. No persistent outputs; prints to stdout.

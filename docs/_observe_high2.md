# _observe_high2.py

**Developer scratch** — not part of the production pipeline.

Second live process/GPU observer for a high-horizon run (variant of
`_observe_high.py`). Hard-codes PID 29284 and polls per-core CPU (`psutil`) and
GPU utilization (`nvidia-smi`).

Hard-codes the PID. No persistent outputs; prints to stdout.

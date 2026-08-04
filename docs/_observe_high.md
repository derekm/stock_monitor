# _observe_high.py

**Developer scratch** — not part of the production pipeline.

Live process/GPU observer for a high-horizon run. Hard-codes PID 12324 and
polls per-core CPU (`psutil`) and GPU utilization (`nvidia-smi`) on an interval.

Hard-codes the PID. No persistent outputs; prints to stdout.

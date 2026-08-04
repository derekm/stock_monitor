# _observe_sweep.py

**Developer scratch** — not part of the production pipeline.

Live process/GPU observer for a parameter sweep run. Hard-codes PID 15960 and
polls per-core CPU (`psutil`) and GPU utilization (`nvidia-smi`) on an interval
while a sweep runs in the background.

Hard-codes the PID. No persistent outputs; prints to stdout.

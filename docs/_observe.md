# _observe.py

**Developer scratch** — not part of the production pipeline.

Live process/GPU observer. Hard-codes a PID (45236) and polls, on an interval,
per-core CPU usage for that process (`psutil`) and GPU utilization via
`nvidia-smi`, printing both so you can watch a training/backfill run's resource
use in real time.

Hard-codes the PID, so it must be edited to match the process you want to watch.
Companion to `_observe2.py`, `_gap_test.py`, `_feed_test.py`. No persistent
outputs.

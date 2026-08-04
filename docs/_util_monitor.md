# _util_monitor.py

**Developer scratch** — not part of the production pipeline.

Long-running resource logger. Polls, for up to 10 hours, per-core CPU usage
(`psutil`) and GPU utilization + memory (`nvidia-smi`) on an interval, writing a
CSV to `/tmp/util_monitor.csv` with columns `t, gpu_util, gpu_mem_mb,
cpu_total, cpu0..cpuN`.

Used to capture a whole training/backfill run's resource profile for later
analysis. Output: `/tmp/util_monitor.csv` (outside the repo).

# _cleanup.py

**Developer scratch** — not part of the production pipeline.

Kills stray local Python worker processes left running by other scratch
benchmarks/probes. It scans `psutil` for `python.exe` processes whose command
line contains any of the markers `_stride_test`, `_cuda_graph`, `_cuda_bench`,
`_ptsteps`, `_observe`, and terminates them.

Used during backfill/forecast benchmarking to clean up orphaned training or
observation processes. Has no inputs or outputs; it is a process-hygiene
helper for development sessions only.

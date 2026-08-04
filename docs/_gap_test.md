# _gap_test.py

**Developer scratch** — not part of the production pipeline.

GPU-occupancy probe for the Granite TTM backfill. It builds full-history
windows, loads the latest global checkpoint and default model, then runs a
background `gpu()` sampler reporting `nvidia-smi` utilization, while a main loop
drives inference — used to find idle gaps where the GPU is not saturated.

Companion to `_observe*.py`. No persistent outputs; prints to stdout.

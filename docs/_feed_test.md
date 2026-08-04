# _feed_test.py

**Developer scratch** — not part of the production pipeline.

Data-feed probe: verifies the window-building and model warm-load path in
`granite_backfill`. It builds full-history windows (AEP), loads the latest
global checkpoint, loads the default Granite model and applies the warm
weights, then samples a GPU/feed loop to confirm the pipeline can keep the
model fed without stalls.

Includes a `gpu()` helper that polls `nvidia-smi` utilization. No persistent
outputs; prints to stdout.

# _p3_iso.py

**Developer scratch** — not part of the production pipeline.

Isolation probe for Pass 3 model construction. It loads the latest global
checkpoint, rebuilds a `TinyTimeMixerConfig` with the Pass-3 baseline shape
(context 512, horizon 96, patch 64, decoder on), constructs the model, and
reports whether the checkpoint's `state_dict` loads (and any strict/loose
mismatches) — used to verify config/checkpoint compatibility before a sweep.

Uses `granite_backfill.gd`. No persistent outputs; prints to stdout.

# _depth_test.py

**Developer scratch** — not part of the production pipeline.

Probe for model depth / context-length behavior in the Granite TTM backfill.
It builds full-history windows, selects AEP, loads the latest global checkpoint
from `GLOBAL_DIR`, loads the default Granite model, and runs scoring/forward
passes at varying depths to observe how prediction quality changes with window
depth.

Uses `granite_backfill.score_windows`. No persistent outputs; prints to stdout.

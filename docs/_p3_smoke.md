# _p3_smoke.py

**Developer scratch** — not part of the production pipeline.

Smoke test for `pass3_sweep.py`. It imports the sweep module, sets `STEPS = 200`,
and runs a small grid (baseline / returns / multi / scratch / lr3e-4) over two
tickers (AEP, NVR) to confirm the sweep runs end-to-end without a long
full-scale job.

Uses `pass3_sweep`. No persistent outputs; prints results to stdout.

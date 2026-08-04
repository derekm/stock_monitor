# _p4_diag2.py

**Developer scratch** — not part of the production pipeline.

Diagnostic for Pass 4 (adjusted-close variant). It imports `pass4` and
`granite_daily`, builds **adjusted** AEP windows (2000, matching the Pass-4
baseline) via `build_windows_p3(..., use_adj=True)`, and trains a few steps
while logging loss/MAPE — to compare adjusted vs unadjusted training dynamics.

Uses `pass4` and `granite_daily`. No persistent outputs; prints to stdout.

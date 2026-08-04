# _p4_diag.py

**Developer scratch** — not part of the production pipeline.

Diagnostic for Pass 4. It imports `pass4`, builds unadjusted AEP windows
(512→96, price, no decoder) via `build_windows_p3`, builds a DataLoader, and
constructs/trains the Pass-3 model for a few steps while logging loss at steps
0/50/100/200 and MAPE — to watch training dynamics and catch divergences early.

Uses `pass4.make_model_p3`. No persistent outputs; prints to stdout.

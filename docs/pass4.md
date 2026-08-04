# pass4.py

**Research/diagnostic** — Pass-4: re-run Passes 2 & 3 on **adjusted** closes,
warm-started from the adjusted global checkpoint (`train_adjusted_full.py`).

## Why it exists (rationale)

Once proper adjusted checkpoints existed (matching the adjusted data
distribution), the warm-start became consistent and the passes comparable. Part
A re-runs the Pass-3 param grid on adjusted closes; Part B runs the Pass-2 regime
sweep on adjusted closes (the old unadjusted-ckpt-on-adjusted-data mismatch had
produced a 715% MAPE artifact). It is the experiment that validated the
adjusted-data path — not part of the production pipeline.

## What it does

- Part A — PASS-3 param grid on adjusted closes (AEP, NVR, FICO), warm-loaded from
  the adjusted global checkpoint; cells also train from IBM scratch as a baseline.
- Part B — PASS-2 regime sweep on adjusted closes (AEP, NVR), warm from the
  adjusted global checkpoint.

## Usage

```bash
python pass4.py        # runs both parts, prints + writes /tmp/pass4_results.json
```

Flags: none (configs are module constants: `P3_GRID`, `P2_WIN`, `P2_STEP`).

## Outputs

- `/tmp/pass4_results.json` — per-cell results. Scratch only; not a repo artifact.
  Prints MAPE / dir-acc / MAE per cell to stdout.

## Related programs

- [pass3_sweep.md](pass3_sweep.md) — Pass-3 baseline
- [train_adjusted_full.md](train_adjusted_full.md) — produces the adjusted checkpoint it warm-starts from
- [granite_backfill.md](granite_backfill.md) / [ttm_backfill.md](ttm_backfill.md)
- [_p4_diag.md](_p4_diag.md) / [_p4_diag2.md](_p4_diag2.md) / [_p4_diag3.md](_p4_diag3.md) (scratch)

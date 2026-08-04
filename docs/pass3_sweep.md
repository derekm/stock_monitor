# pass3_sweep.py

**Research/diagnostic** — Granite TTM parameter sweep (Pass 3).

## Why it exists (rationale)

A controlled experiment to find good TTM hyper-parameters. It holds windowing
constant (fixed 200 windows, effective stride ~3, matching the production
backfill) and varies one axis at a time: `context` (256/512/1024), `horizon`
(32/96/240), `patch_length` (8/16/32), `use_decoder` (True/False), `objective`
(price/returns), `multivariate` (True/False), on cleaned adjusted-history data.
Sample tickers: AEP (low-vol), NVR (high-vol), FICO (mid). Not part of the
production pipeline.

## Usage

```bash
python pass3_sweep.py        # runs the full grid, prints + writes /tmp/pass3.json
```

Flags: none (constants at top: `STEPS = 6000`, `TICKERS`, `GRID`, `DEFAULTS`).

## Outputs

- `/tmp/pass3.json` — per-cell results (MAPE, dir-acc, MAE, config). Scratch only;
  not a repo artifact. Prints a summary to stdout.

## Related programs

- [pass4.md](pass4.md) — Pass-4 (adjusted closes) follow-up
- [granite_backfill.md](granite_backfill.md) / [ttm_backfill.md](ttm_backfill.md) — production training
- [_p3_debug.md](_p3_debug.md) / [_p3_iso.md](_p3_iso.md) / [_p3_smoke.md](_p3_smoke.md) (scratch)

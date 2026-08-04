# vol_steps.py

**Research/diagnostic** — volatility-based per-ticker training-step allocator.

## Why it exists (rationale)

From the Pass-1/2 sweeps: low-vol names (e.g. AEP) plateau ~9k steps while
high-vol names (e.g. NVR) are still improving at 12k. So the step budget should
scale *with* volatility (more volatile ⇒ harder to fit ⇒ more steps). This
computes a per-ticker step allocation from realized vol, as a guide for the
backfill config — not a production pipeline step.

## Usage

```bash
python vol_steps.py        # prints per-ticker step allocation
```

Flags: minimal (reads `daily_prices.parquet`; see source for any sub-commands).
Prints to stdout.

## Outputs

None written to disk (prints the allocation). Consumed as a config guide for
`ttm_backfill` / `granite_backfill`.

## Related programs

- [ttm_backfill.md](ttm_backfill.md) / [granite_backfill.md](granite_backfill.md) — step budgets
- [_ptsteps_high2.md](_ptsteps_high2.md) (scratch)
- [vol_target.md](vol_target.md)

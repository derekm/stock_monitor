# train_adjusted_full.py

**Thin config wrapper** over the factored library `ttm_backfill` — full historical
pre-training on **adjusted** closes.

## Why it exists (rationale)

`granite_daily.py` (and the production forecast path) need a well-trained
*adjusted* checkpoint to warm-start from. This builds
`ttm_backfill.adjusted_backfill_config()` (mirroring the historical default:
steps=150, chunk=90) and runs `ttm_backfill.run_backfill`, writing the adjusted
global checkpoint that `pass4` / `granite_daily` / `forecast_granite` consume.

## Usage

```bash
python train_adjusted_full.py --steps 150 --chunk 90
python train_adjusted_full.py --tickers AEP,NVR --batch 4
```

Flags: `--steps` (default 150), `--chunk` (default 90), `--batch`, `--tickers`.

## Outputs

None written directly (training writes dated checkpoints under `checkpoints/` via
`ttm_backfill`). See [ttm_backfill.md](ttm_backfill.md).

## Related programs

- [ttm_backfill.md](ttm_backfill.md) — the real implementation
- [pass4.md](pass4.md) — warm-starts from this checkpoint
- [granite_daily.md](granite_daily.md) / [forecast_granite.md](forecast_granite.md)
- [backfill_historical.md](backfill_historical.md) — adjusted history source

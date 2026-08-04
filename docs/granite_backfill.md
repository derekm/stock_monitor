# granite_backfill.py

**Thin backward-compatible shim** over the factored library `ttm_backfill.py`.

## Why it exists (rationale)

The TTM training logic was extracted into `ttm_backfill.py` (config/callback
library with `BackfillConfig` / `RegimeConfig`). This file stays only so older
imports and muscle-memory keep working: it re-exports the public API
(`build_full_history_windows`, `train_windows`, `train_aggregate`,
`train_checkpoint`, `score_windows`, `per_ticker_plan`, `_clean_price_frame`,
`run`, `coverage_report`, `main`) and delegates `run()` to
`ttm_backfill.run_backfill` with the historical default config (steps=150,
chunk=90). For new model regimes, call `ttm_backfill` directly.

## Usage

```bash
python granite_backfill.py run --tickers AEP,NVR --steps 150 --chunk 90
python granite_backfill.py run --compare
python granite_backfill.py coverage
```

Flags: `--tickers`, `--steps` (default 150), `--chunk` (default 90), `--batch`,
`--compare`, `--tickers`.

## Outputs

None written directly (training writes dated checkpoints under `checkpoints/`
via `ttm_backfill`). See [ttm_backfill.md](ttm_backfill.md).

## Related programs

- [ttm_backfill.md](ttm_backfill.md) — the real implementation
- [granite_daily.md](granite_daily.md) — consumes the checkpoint for daily forecasts
- [forecast_granite.md](forecast_granite.md)
- [backfill_historical.md](backfill_historical.md) — history source

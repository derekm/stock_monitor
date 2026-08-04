# ttm_backfill.py

Config + callback driven library for pre-training Granite TinyTimeMixer (TTM)
over the full `daily_prices` history for an arbitrary ticker set, at arbitrary
model regimes (global aggregate, proxy-padded aggregate, per-ticker), with
adjusted / unadjusted closes.

## Why it exists (rationale)

This is the factored-out core that `granite_backfill.py` used to inline. It
exposes `BackfillConfig` / `RegimeConfig` / `Callbacks` so any model regime
(global / aggregate / per-ticker, adj/unadj) can be set up explicitly, and
`run_backfill()` does the training — writing dated checkpoints under
`checkpoints/`. `granite_backfill` and `train_adjusted_full` are thin shims over
it.

## Usage

```bash
python ttm_backfill.py run --tickers AEP,NVR --steps 150 --chunk 90
python ttm_backfill.py run --adjusted --tickers AEP
python ttm_backfill.py coverage
```

Sub-commands: `run`, `coverage`. Flags: `--tickers`, `--steps` (default 150),
`--chunk`, `--batch`, `--adjusted`, `--compare`.

## Key API

- `adjusted_backfill_config(steps=150, batch=None, …)` — adjusted-history config
- `run_backfill(cfg, prices=None)` — trains, returns a summary dict

## Outputs

Dated checkpoints under `checkpoints/` (per-ticker / aggregate / global).

## Related programs

- [granite_backfill.md](granite_backfill.md) / [train_adjusted_full.md](train_adjusted_full.md) — shims
- [granite_daily.md](granite_daily.md) / [forecast_granite.md](forecast_granite.md) — consumers
- [ttm_features.md](ttm_features.md) — builds the panels it trains on
- [backfill_historical.md](backfill_historical.md)

# granite_daily.py

The "own the forecasts" daily pipeline: refresh → continual retrain → forecast →
score, for the Granite TTM.

## Why it exists (rationale)

Zero-shot Granite forecasts don't compound. This is the production daily loop:
append prior-day actuals to a persistent per-ticker cache, take a few gradient
steps warm-started from yesterday's checkpoint (so learning compounds), save a
dated checkpoint, forecast the next 96 trading days for every covered ticker
(writing `forecasts_granite.parquet`), and score today's forecast against
actuals since. It is what makes the forecast tab live.

## Usage

```bash
python granite_daily.py run --retrain            # full refresh+retrain+forecast+score
python granite_daily.py run --tickers AEP,NVR --limit 2 --steps 50
python granite_daily.py status                   # cache/checkpoint state
python granite_daily.py score                    # re-score stored forecasts
```

Subcommands: `run`, `status`, `score`. Flags: `--tickers`, `--retrain`,
`--limit`, `--steps` (default `STEPS_PER_DAY`), `--batch`.

## Outputs

- `granite_series_cache.parquet` — persistent per-ticker realized series
- `forecasts_granite.parquet` — daily forecasts (overwrites each run)
- `granite_ckpts/` — dated checkpoints (per-ticker / bucket)
- `granite_accuracy.json` — rolling forecast accuracy

(Schema families: forecast_anomaly / aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [granite_backfill.md](granite_backfill.md) / [ttm_backfill.md](ttm_backfill.md) — pretraining
- [forecast_granite.md](forecast_granite.md) — one-shot forecast/backtest
- [granite_service.md](granite_service.md) — serves the latest forecast
- [sp500_constituents.md](sp500_constituents.md) — coverage list

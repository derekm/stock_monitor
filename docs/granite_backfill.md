# granite_backfill.py — historical TTM pre-training

> **Quick start & day-to-day usage is here.** For the full config/callback API, see
> [ttm_backfill.md](ttm_backfill.md).

`granite_backfill.py` pre-trains **Granite TinyTimeMixer (TTM-r2)** over the full
`daily_prices.parquet` history for every covered ticker, so the daily
[`granite_daily.py`](../granite_daily.py) forecaster runs start from well-trained models
instead of the cold IBM zero-shot base.

It is now a **thin backward-compatible shim** over the factored library
[`ttm_backfill.py`](ttm_backfill.md). All training and windowing logic lives in
`ttm_backfill`; this module re-exports the public API older scripts still import and
delegates `run()` to `ttm_backfill.run_backfill` with the historical default config.

```python
from granite_backfill import (
    build_full_history_windows, train_windows, train_aggregate,
    score_windows, per_ticker_plan, _clean_price_frame,
    run, coverage_report, gd,
)
# train_windows / train_aggregate are aliases for ttm_backfill.train_checkpoint
```

## Why pre-train at all

The daily `granite_daily.py` forecast loop is cheaper and more accurate when it warm-starts
from a model that has already seen a ticker's whole price history. `granite_backfill`
produces those warm-start checkpoints in three regimes:

| regime | applies to | warm-starts from | output dir |
|--------|-----------|------------------|------------|
| `global` | all tickers | IBM pretrained | `granite_ckpts/global/` |
| `padded` | short tickers (proxy-padded) | global | `granite_ckpts/padded/` |
| `per_ticker` | full tickers | global (shorts ← padded) | `granite_ckpts/per_ticker/<TICKER>/` |

This is the **identical recipe** the original `granite_backfill.run()` produced, now driven
by `default_backfill_config()`.

## Default backfill (unadjusted)

```bash
python -m granite_backfill run --steps 150 --batch 16
# or, with a ticker subset + comparison report:
python -m granite_backfill run --tickers AEP,NVR,FICO --steps 150 --compare
```

`--steps` defaults to 150 (fast, enough for warm-start and relative signal). Raise it
(e.g. 6000) for production-grade per-ticker checkpoints.

## Adjusted backfill (controlled comparison)

```bash
python train_adjusted_full.py --steps 150 --batch 16
```

`train_adjusted_full.py` is a one-function wrapper: it calls
`adjusted_backfill_config()` (same regime structure, `adj_close`, no-adjust tickers
excluded) and runs `run_backfill`. The **only** variable vs the unadjusted run is the price
source, so the two checkpoint families are directly comparable. Output dirs:
`granite_ckpts/adjusted_global` / `adjusted_padded` / `adjusted_per_ticker`.

## Direct adjusted-vs-unadjusted comparison

```bash
python -m ttm_backfill cmp-adj-unadj --tickers AEP,NVR,FICO --steps 150
```

Trains both an adjusted and an unadjusted per-ticker regime on the **same windows** and
reports MAPE for each. Conclusion (150-step): adjusted closes give **no meaningful
improvement** — NVR is marginally better (25.73% vs 27.31% MAPE), AEP/FICO essentially
identical. At 150 steps this is a *relative* signal; absolute MAPE is not production-grade.

## Coverage report

```bash
python -m granite_backfill coverage
```

Reports per-ticker window readiness: which tickers have a full 512-context history vs which
are short and require proxy-padding (see `window_padding.py`). Useful before a big backfill
to confirm coverage.

## Regime sweeps (arbitrary model regimes)

To explore context length / horizon / learning rate at the per-ticker level, use
`ttm_backfill.sweep_regimes()` (documented in [ttm_backfill.md](ttm_backfill.md)). Each
regime is config, not a copy-pasted loop:

```python
import ttm_backfill as t
from ttm_backfill import TrainConfig, DataConfig
t.sweep_regimes(
    tickers=["AEP"], base_regime="per_ticker",
    base_ckpt_dir=t.CKPT_DIR/"adjusted_per_ticker",
    regimes=[
        ("baseline", TrainConfig(steps=6000), None, None),
        ("hor32",    TrainConfig(steps=6000), DataConfig(horizon=32),
                     {"context_length":512,"prediction_length":32,
                      "patch_length":16,"use_decoder":True}),
        ("lr3e-4",   TrainConfig(steps=6000, lr=3e-4), None, None),
    ], use_adj=True)
```

## Outputs

| path | contents |
|------|----------|
| `granite_ckpts/global/granite_ttm_tuned_<date>.pt` | global aggregate ckpt |
| `granite_ckpts/padded/granite_ttm_tuned_<date>.pt` | padded aggregate ckpt |
| `granite_ckpts/per_ticker/<TICKER>/<TICKER>_tuned_<date>.pt` | per-ticker ckpts |
| `granite_ckpts/adjusted_*/...` | adjusted equivalents |
| `ttm_backfill_compare.jsonl` | global vs per-ticker vs padded MAE rows |

> Checkpoints are **not** committed to git (see `.gitignore`). They are regenerable and
> large.

## Data hygiene (shared with the library)

`ttm_backfill._clean_price_frame` is what makes training stable. It:

1. drops duplicate `(ticker, date)` pulls (yfinance can disagree between pulls),
2. optionally clips to the last N trading days,
3. drops impossible single-day moves (`|adj logret| > 0.3`, forward **and** backward),
4. collapses remaining `(ticker, date)` conflicts by mean,

and returns the chosen column (`adj_close` when `use_adj`, else `close`) as `close`.

## Migration note

Older scripts that `import granite_backfill` keep working unchanged — the shim re-exports
`build_full_history_windows`, `train_windows`, `train_aggregate`, `score_windows`,
`per_ticker_plan`, `_clean_price_frame`, `run`, `coverage_report`, `main`, and the module
constants. New code that needs **arbitrary regimes** (global + per-ticker) should import
`ttm_backfill` directly and build a `BackfillConfig` (see
[ttm_backfill.md](ttm_backfill.md)).

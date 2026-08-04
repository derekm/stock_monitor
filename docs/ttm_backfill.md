# ttm_backfill.py — Granite TTM pre-training library

> **Library reference.** For the high-level "why" and the day-to-day commands, see
> [granite_backfill.md](granite_backfill.md). This file documents the config +
> callback API in `ttm_backfill.py` in detail.

`ttm_backfill.py` is the factored core that trains **Granite TinyTimeMixer (TTM-r2)**
over the full `daily_prices.parquet` history for an arbitrary set of tickers, at
arbitrary model regimes (global aggregate, proxy-padded aggregate, per-ticker), with
arbitrary price sources (adjusted / unadjusted closes).

It replaced the logic that `granite_backfill.py` used to inline. It is **config +
callback driven** (mirroring how `fisher_index.py` generalizes index construction),
so arbitrary model regimes can be composed at the global and per-ticker level **without
editing the training loop**.

## Why it exists

The daily `granite_daily.py` forecaster runs are far cheaper and more accurate when they
start from a model that has already seen the ticker's whole history, instead of the
cold IBM zero-shot base. `ttm_backfill` produces those warm-start checkpoints. The two
things `granite_daily` consumes are:

- **per-ticker checkpoints** — one tuned model per ticker (`granite_ckpts/<set>/<ticker>/<ticker>_tuned_<date>.pt`)
- **global / padded aggregates** — a single model trained across many tickers (used as
  the warm-start seed and as a fallback forecaster)

## Core concepts

| Concept | Role |
|---------|------|
| `DataConfig` | What data to use: price column (adj/unadj), cleaning, which tickers, context/horizon, window sampling, short-ticker padding, exclusions. |
| `TrainConfig` | How to train ONE checkpoint: steps, batch, lr, optimizer, grad-clip, dtype, warm-start source. |
| `RegimeConfig` | A named training *regime*: a `TrainConfig` + which tickers it applies to (`full`/`short`/`all`) + where to save + which prior regime to warm-start from + optional per-regime `data`/`model_config` overrides. |
| `BackfillConfig` | A full job: one `DataConfig` + an ordered list of `RegimeConfig` + an optional comparison step + callbacks. |
| `Callbacks` | Hooks fired during the run (window build, per-train-step, per-ticker, per-regime, comparison) so callers can observe / log / abort without touching the engine. |

A *regime* at the per-ticker level means: a per-ticker checkpoint family trained with a
given `TrainConfig`. Multiple regimes can share the same global/padded base but differ in
`lr` / `steps` / optimizer — which is exactly what lets the pass-3/4 parameter sweeps be
expressed as **config instead of copy-pasted loops**.

## `DataConfig`

```python
@dataclass
class DataConfig:
    use_adj: bool = False                       # True -> train on adj_close
    recent_trading_days: Optional[int] = None   # None = full history
    max_daily_logret: float = 0.3               # drop impossible single-day moves
    context: int = 512                           # TTM-r2 FIXED context; never shrunk
    horizon: int = 96
    max_windows_per_ticker: int = 200           # linspace cap (matches old default)
    pad_short: bool = True                       # proxy-pad short tickers
    exclude_tickers: list[str] = []              # e.g. no-adj tickers
    tickers: Optional[list[str]] = None         # None = all (after exclusions)
    out_dir: Path = CKPT_DIR
```

### Data hygiene (`_clean_price_frame`)

The raw `daily_prices.parquet` has defects that corrupt TTM training. The cleaner fixes
them in order:

1. **Duplicate pulls** — the same `(ticker, date)` can appear multiple times, sometimes
   with *conflicting* close (two yfinance pulls disagree). Dropped by `drop_duplicates`.
2. **Unadjusted long history** — pre-split prices span ~100× scale. With `adj_close`
   captured, the adjusted series is stationary across decades, so training uses `adj_close`
   when `use_adj=True`.
3. **Adjustment errors** — split/dividend bugs inject impossible single-day moves
   (e.g. AEP `131.7 → 33.7 → 131.4` in one day). Rows whose adjacent `adj_close`
   log-return exceeds `max_daily_logret` (default 0.3) are dropped, in **both** forward
   and backward direction (so a spike doesn't leave a neighbor orphaned).
4. **Conflicts** — remaining `(ticker, date)` duplicates are collapsed by mean.

The chosen price column is returned as `close` so downstream code is source-agnostic.

### No-adjustment tickers (`no_adj_tickers`)

`tickers whose adj_close ≈ close` (relative mean diff < 1e-4) have no real adjustment
data. If included in *adjusted* training they would silently train on a flat (unadjusted)
series masquerading as adjusted. `adjusted_backfill_config()` auto-excludes them (≈95
tickers, e.g. ABNB, AMD, AMZN). The comparison is therefore done on the *intersection*
of tickers that actually have adjustment history.

## `TrainConfig`

```python
@dataclass
class TrainConfig:
    steps: int = 150
    batch: int = gd.BATCH
    lr: float = gd.LR                 # 1e-4
    optimizer: str = "adamw"          # adamw | adam | sgd
    grad_clip: float = 5.0
    dtype: str = "float32"
    compare: bool = False
```

`gd.LR`, `gd.BATCH` come from `granite_daily` (the shared TTM defaults). NaN/inf loss
aborts the training loop for that seed (printed, not raised) so one bad window can't kill
a sweep. Gradient norm is clipped to `grad_clip`.

## `RegimeConfig`

```python
@dataclass
class RegimeConfig:
    name: str
    applies_to: str = "full"            # full | short | all
    out_dir: Path = PER_TICKER_DIR
    warm_from: Optional[str] = "pretrained"
    train: TrainConfig = TrainConfig()
    kind: str = "per_ticker"            # per_ticker | aggregate
    data: Optional[DataConfig] = None   # per-regime window override (ctx/hor/price)
    model_config: Optional[dict] = None # per-regime TTM arch override
    agg_windows_per_ticker: int = 200   # aggregate subsample cap
```

`warm_from` names a prior regime to warm-start from:

| value | meaning |
|-------|---------|
| `"pretrained"` | IBM zero-shot base |
| `"global"` | the global aggregate regime |
| `"padded"` | the padded aggregate regime |
| `"self"` | this regime's own latest ckpt (incremental) |
| `None` | same as `pretrained` |

### Two first-class sweep dimensions

- **`data`** (a `DataConfig`) — overrides context/horizon/**price source** for this regime
  only. Lets horizon and adjusted-vs-unadjusted become per-regime parameters without
  touching the job-wide config.
- **`model_config`** (a dict `{"context_length", "prediction_length", "patch_length",
  "use_decoder"}`) — builds a **fresh** TTM model with those architecture hyperparameters
  instead of loading the default IBM ckpt.

> **Important:** `context_length` / `prediction_length` are *model-architecture*
> hyperparameters in TTM — the output head shape depends on the horizon. So a **horizon
> sweep must construct a fresh model** (via `TinyTimeMixerConfig`) rather than only
> reshaping windows. When `model_config` is set, cross-architecture warm-start is disabled
> (`warm_sd=None`); the regime trains from the IBM base for that architecture.

## `Callbacks`

All optional. `on_ticker` / `on_regime` may return `False` to abort the run.

| hook | called with |
|------|-------------|
| `on_window_build(n_tickers, secs)` | after windows are built |
| `on_train_step({step, loss, ...meta})` | every training step |
| `on_train_end({name, steps, n, secs, out_path})` | after a checkpoint trains |
| `on_ticker({i, n, tk, wins, secs})` | per ticker (may abort) |
| `on_regime({regime, phase})` | regime start/done (may abort) |
| `on_compare(list[rows])` | comparison table rows |
| `on_log(message)` | arbitrary log line |

A natural use: wire `on_train_step` to the live CPU/GPU monitor so every regime logs
utilization per step.

## `BackfillConfig` + `run_backfill()`

```python
@dataclass
class BackfillConfig:
    data: DataConfig
    regimes: list[RegimeConfig]
    compare: bool = True
    callbacks: Callbacks
    compare_log: Path = HERE / "ttm_backfill_compare.jsonl"
```

`run_backfill(cfg)` executes three phases:

1. **Build windows once** (`build_windows`) — cached per ticker. Full tickers get raw
   stride-1 rolling windows, linspace-capped at `max_windows_per_ticker` (default 200).
   Short tickers get a proxy-padded context head (sector/market proxy from `window_padding`)
   + their own target.
2. **Run regimes in order** — aggregate regimes (`applies_to="all"`) train one ckpt over
   all applicable windows (subsampled per ticker for RAM); per-ticker regimes train one
   ckpt per applicable ticker. Warm-start chains follow `regime.warm_from`.
3. **Optional comparison** — score every per-ticker under global vs per-ticker vs padded
   (MAE), append rows to `compare_log` (JSONL), print a per-ticker table.

The default `default_backfill_config()` reproduces the historical
`granite_backfill.run()` exactly: 3 regimes — `global` (all, ← pretrained) → `padded`
(short, ← global) → `per_ticker` (full, ← global; shorts warm from padded). Output dirs:
`granite_ckpts/global`, `granite_ckpts/padded`, `granite_ckpts/per_ticker`.

## `adjusted_backfill_config()`

Identical recipe to `default_backfill_config()` **but on `adj_close`**, with the no-adjust
tickers auto-excluded, and output dirs `granite_ckpts/adjusted_global` /
`adjusted_padded` / `adjusted_per_ticker`. The **only** variable vs the unadjusted run is
the price source → the two checkpoint families are directly comparable (controlled
variable).

## `compare_adj_unadj()`

Trains (undertrained, `steps`) **both** an adjusted and an unadjusted per-ticker regime for
the same tickers, on the **same windows** (sampled identically via `DataConfig`), then
reports MAPE/MAE for each on identical eval windows. The only variable is the price source
→ a clean controlled test of whether adjusted closes help.

Empirical result (150-step undertrained, AEP/NVR/FICO): adjusted closes give **no
meaningful improvement** (NVR marginally better at 25.73% vs 27.31% MAPE; AEP/FICO
essentially identical). At 150 steps this is a *relative* signal only — undertrained
checkpoints rank regimes correctly but absolute MAPE is not production-grade.

## `sweep_regimes()` — config-driven parameter sweeps

```python
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

Each regime is a `(name, TrainConfig[, DataConfig][, model_config])` tuple. Every regime
warm-starts from `base_regime`'s latest ckpt and saves into its own `granite_ckpts/sweeps/<name>/`
subdir. This is the config-driven replacement for the copy-pasted pass-3/4 grids — no loop
duplication, and `context length`, `horizon`, and `learning rate` are all first-class
sweep dimensions.

### Handling NaN/inf

The training loop watches `loss`; if it goes non-finite it aborts that seed and prints
`[NaN/inf loss at step N; aborting]`. Gradient norm is clipped (`grad_clip=5.0`). This
keeps sweeps robust when a single window has a residual bad value after cleaning.

## Outputs

| path | contents |
|------|----------|
| `granite_ckpts/global/granite_ttm_tuned_<date>.pt` | global aggregate ckpt |
| `granite_ckpts/padded/granite_ttm_tuned_<date>.pt` | padded aggregate ckpt |
| `granite_ckpts/per_ticker/<TICKER>/<TICKER>_tuned_<date>.pt` | per-ticker ckpts |
| `granite_ckpts/adjusted_*/...` | adjusted equivalents |
| `granite_ckpts/sweeps/<name>/<TICKER>/...` | sweep regime ckpts |
| `ttm_backfill_compare.jsonl` | global vs per-ticker vs padded MAE rows |

> Checkpoints are **not** committed to git (see `.gitignore`). They are regenerable and
> large.

## CLI

```bash
python -m ttm_backfill run [--tickers AEP,NVR] [--steps 150] [--batch 16] [--use-adj] [--compare]
python -m ttm_backfill coverage          # per-ticker window readiness report
python -m ttm_backfill cmp-adj-unadj --tickers AEP,NVR,FICO [--steps 150]
```

`run` builds the default (or `--use-adj`) config and calls `run_backfill`. `coverage`
reports full-512-context vs short/padded readiness per ticker. `cmp-adj-unadj` runs the
controlled adjusted-vs-unadjusted comparison.

## Example: custom regime appended to the default job

```python
import ttm_backfill as t
from ttm_backfill import RegimeConfig, TrainConfig

cfg = t.default_backfill_config(steps=150, batch=16)
cfg.regimes.append(RegimeConfig(
    name="per_ticker_lr3e-4", applies_to="full",
    out_dir=cfg.data.out_dir / "per_ticker_lr3e-4",
    warm_from="per_ticker", train=TrainConfig(steps=150, lr=3e-4)))
t.run_backfill(cfg)
```

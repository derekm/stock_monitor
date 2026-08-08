# AGENTS.md — for agents operating on / answering questions about stock_monitor

This repo is a **personal portfolio intelligence stack** (Python + DuckDB + a static/DuckDB-Wasm dashboard) owned by a GPU/CUDA-fluent quant. It is *not* a generic library — it is a growing collection of 130 scripts that read/write a small set of canonical parquet tables.

## Mental model

- **Inputs are few, outputs are many.** Nearly everything reads `daily_prices.parquet`, `fundamentals.parquet`, `monitored_stocks.parquet`, `portfolio_holdings.parquet`, `trades.parquet`, `sector_prices.parquet`, `sp500_constituents.parquet`, `alerts_config.parquet`, `earnings_calendar.parquet`. Everything else is a derived CSV/parquet.
- **One orchestrator to rule them:** prefer `run_daily_automation.py` (or the dashboard's `analytics_service` → `/run/all-daily`) over calling individual scripts. Individual scripts exist for targeted re-runs and research.
- **Dashboard = 4 services + static site**, all launched by `./start_dashboard.sh` (granite_service :5055, pipeline_service :5056, analytics_service :8767, static :8765). Ctrl+C stops all.
- **Forecasts are stateful:** they need pretrained checkpoints under `checkpoints/` (Git-ignored, large). `granite_backfill.py`/`ttm_backfill.py` pretrain; `granite_daily.py` runs daily; `forecast_granite.py` emits `forecasts_granite.csv/.parquet`. **Regime-selected serving** (`regime_serving.py` + `checkpoints/regime/`) ensembles the current HMM regime's model when available.
- **Signals aggregate honestly:** five families (preferred/peer/cross/pairs/earnings) + technical/options/revisions/sentiment merge in `signal_aggregator.py` with OOS-IC-derived weights; the composite feeds `buy_candidates.py`, `shadow_book.py`, and the regime-gated forecast annotations.
- **The Taleb layer is the uncertainty audit:** `tail_index` → `ergodicity_ruin`; `gap_risk` + `tail_index` → `fragility_screen` → `barbell_check`; `aggregate` + `preferred` → `hidden_optionality_audit` (decision flip rates). The audit's fixes live in `buy_candidates.py`: soft stress posterior + noise-convolved driver expectations (see Key facts).

## If asked to "run the analytics"

1. Check data freshness first: if `daily_prices.parquet` looks stale, `python update_prices.py --fetch --days 5`.
2. Run `python run_daily_automation.py` (or `/run/all-daily` via the dashboard). Use `--only <jobs>` for a subset (32 valid jobs: `hmm, rebalance, preferred, inclusion, stress, crisis, factor_rot, risk_enrich, rolling, rolling_corr, tail_hedge, allpairs, fund_snap, screen_bt, dupont, growth, peer, earnings, pairs, cross, aggregate, technical, econ_cal, est_rev, shadow, taleb_tail, taleb_gap, taleb_ergodic, taleb_fragility, taleb_barbell, taleb_optionality, export`).
3. Refresh the dashboard: `python export_dashboard_data.py` (rewrites `dashboard_data/data.json`, 198 resources).
4. If the user wants forecasts and `forecasts_granite.parquet` is missing: `python granite_backfill.py` → `python granite_daily.py` → `python forecast_granite.py forecast --index portfolio --from-first-trade --horizon 10`.
5. If the user wants regime-selected forecasts and `checkpoints/regime/` is empty: `python pass6.py --tickers <list> --ckpt-dir checkpoints/regime` (GPU, long), then verify with `python regime_serving.py` (prints the serving plan). Tickers without coverage keep the general model — that is the intended degradation.

## If asked to "add / change a script or output"

- **Don't hand-edit the base parquet tables.** Use the dedicated writers (`manage_stocks.py`, `update_fundamentals.py`, `update_prices.py`, `manage_alerts.py`).
- **Add a new output?** Update `docs/SCHEMAS.md` (the output catalog) and the script's `docs/<script>.md` (Outputs + Related). Keep schema families deduplicated — don't redefine a column list already in a family.
- **New script?** Add `docs/<script>.md` with: one-line description, "Why it exists (rationale)", Usage (`--help` for flags; standard `cli_common` flags apply), Outputs (link to SCHEMAS family), Related programs.
- **Naming:** outputs go to `DATA_DIR` (the repo root, where the parquets live), not cwd. Use `DATA_DIR / "name.csv"`.
- **New daily job?** Add it to `run_daily_automation.py` JOBS (cmd + timeout) and DEPS (dependency set), and register its outputs in `export_dashboard_data.py` TABLES so the dashboard exposes them.
- **New diagram?** Mermaid sources live in `docs/diagrams/*.mmd`; re-render with `python render_mermaid.py` (kroki.io, dark theme).

## Key facts an agent must not get wrong

- `analytics_service.py` script default port is **8765**; `start_dashboard.sh` overrides it to **8767** (so the static server keeps 8765). Don't tell a user the service is on 8765 when launched via the script.
- `portfolio_optimization.py` takes a uniform `--name-cap` (default 5%) applied to **all** names; there is no per-ticker special-casing. Use that flag rather than any removed ticker-specific cap.
- `fisher_index.py` writes `fisher_indexes.csv`, `fisher_indexes.parquet`, **and** `fisher_rate_decomposition.csv`. `run_fisher_duckdb.py` is the DuckDB system-of-record variant (`fisher_indexes_duckdb.csv/.parquet`).
- `stress_dual_pass.py` pass-counts are **data-dependent** (recomputed against current fundamentals) — never quote a fixed count as if it were a constant.
- Checkpoints under `checkpoints/` are large and Git-ignored — never delete them mid-run; never commit them.
- `daily_prices.parquet` is stored in **Git LFS** (exceeds GitHub's 100 MB limit). Use `git lfs` for any push involving it.
- **Granite config constants live in `granite_config.py`** (a leaf module: `DEFAULT_MODEL`, `CONTEXT=512`, `HORIZON=96`). `granite_daily.py` re-exports them; `window_padding.py`/`forecast_granite.py` import from `granite_config` — **never** import them from `granite_daily` (circular-import trap: granite_daily's chain imports window_padding).
- **Regime-model checkpoints** (`checkpoints/regime/*.pt`) are named `<TICKER>__<regime>__<steps>__<lr>.pt` and carry `n_channels` (1=close-only, 3=close+return+vol20) + `trained_on`. The serving side (`regime_serving.py`) matches by current HMM regime + `regime_model_best.csv` config; mismatched names silently fall back to the general model (`no_checkpoint` reason) — verify with `python regime_serving.py`.
- **pass5/6/7/8 share machinery**: `pass6.py` imports `tag_windows`/`temporal_split` from itself and `windows_with_dates` from `regime_forecast.py`; `pass7.py` imports `train_regime_model` from `pass6`; `pass8.py` imports the same plus the pass6 hooks (`_CUSTOM_BASE_CKPT` for custom bases, `_exog_channel`/`_load_event_dates` for the exog channel). The resume dedupe key is the full cell identity (ticker, regime, split_frac, steps, cap, lr, composition) — not the combined arm string. pass6's `--head-only` freezes the backbone (TTM-paper mode); `--exog` adds the calendar-event channel; `--rpt` probes the base and falls back truthfully (RPT needs a base PRE-TRAINED with `num_patches=9` — see pass8).
- `signal_aggregator.py` weights are **per-regime OOS-IC-derived** (current HMM regime picks the weights); the composite is consumed by `buy_candidates.py` and `shadow_book.py`.
- **buy_candidates decisions are noise-robust (don't "simplify" them back to thresholds).** The stress haircut reads the HMM **posterior** (`regime_stress_prob()` — score −= 0.08·p(stress)); every numeric driver's contribution is the **noise-convolved expectation** `_step_expectation(x, sig, baseline, steps)` over its estimation error (`_est_error` = cross-sectional std/4). The `*STEPS` configs (`MOMENTUM_STEPS`, `FACTOR_STEPS`, `COMPOSITE_STEPS`, `RESID_MOM_STEPS`, `LIQUIDITY_STEPS`, `SKEW_STEPS`) are the single source of truth for the thresholds; `sig=0` reproduces the old exact thresholds. This came from the hidden-optionality audit (hard regime cliff flipped 28.4% of decisions; momentum 6.8% → ~6% with the knife-edges gone). `hidden_optionality_audit.py` (the `taleb_optionality` job) re-measures flip rates daily.
- **Forecast uncertainty is Student-t, not Gaussian.** `forecast_ttm_mc_dropout` returns `(mean, std, nu)` — ν from the MC sample kurtosis (kurt_t = 6/(ν−4), clamped [4,30]); emitted as `forecast_nu`. `--epistemic-error EPS` widens std by √(1+EPS²) (the Forecasting-Paradox 50/50 σ(1±EPS) scale mixture) and thins ν accordingly.
- **pass8 RPT convention:** our own RPT base uses `num_patches=9` (freq token = 9th patch slot) and **daily `freq_token=8`** — the granite-tsfm canonical mapping (time_series_preprocessor.py `DEFAULT_FREQUENCY_MAPPING`: min=1, h/H=7, d/D=8, W=9). Do NOT change the token to 2 — that's Google TimesFM's convention, not IBM's. `pass6 --rpt` probes the base and falls back truthfully (`rpt=False` in the result) when the base wasn't RPT-pretrained.
- **The rebalance calendar uses the soft stress band** `1 − 0.5·p(stress)` (HMM posterior via `stress_prob_on`) — `stress_prob` column added; no hard half-band cliff.

## Documentation layout

- `docs/SYSTEM_OVERVIEW.md` — investment thesis, screens, risk, regime tools, S&P-500 design.
- `docs/SYSTEM_ORCHESTRATION.md` — data flow, service map, "what an agent should know before running analytics."
- `docs/SCHEMAS.md` — every output file → script → schema family (single catalog).
- `docs/<script>.md` — one doc per script (description, rationale, outputs, cross-links).
- `README.md` — quick start + "How to update / re-run."
- `docs/diagrams/` — framework PNGs + Mermaid sources (re-render via `render_mermaid.py`).
- [GLOSSARY.md](https://github.com/derekm/stockmagic/blob/master/GLOSSARY.md) — cross-repo acronym dictionary (lives in the parent repo).

## Gotchas from history (so you don't repeat them)

- The repo was largely generated by an earlier AI with one-off hacks and a partly-misunderstood schema (e.g. ADRs/ETFs in `sp500_member`, `daily_prices` column is `date` not `trade_date`). Trust the data, but verify before asserting.
- The `ttm_backfill.py` / `granite_backfill.py` adj-close pipeline: `backfill_historical.py` captures `adj_close`; `_clean_price_frame` dedups and drops impossible moves; yfinance can retroactively re-resolve `adj_close` after corporate actions, so a rescale detector exists. If a user reports "prices look wrong after a split," suspect adj-close rescale, not a bug in the loader.
- `torch.compile` / inductor **fails on Windows** (no TritLW wheels). True parallelism is `multiprocessing` (separate processes, each with its own model copy). Don't suggest `torch.compile` here.
- When the user reports a performance bottleneck, measure live per-core CPU + `nvidia-smi` GPU util over a long window before asserting "GPU-bound" or "0% CPU." Perf claims need live measurement, not assertions.
- **Multi-channel TTM** (`pass6.py --channels 3`, close+return+vol20) expands the pretrained 1-channel model via `AutoConfig(num_input_channels=3)` + `from_pretrained(ignore_mismatched_sizes=True)` — not `resize_token_embeddings` (that's the vocab API, not the channel API, and raises `NotImplementedError` here). The serving side builds the same channels from a checkpoint's `n_channels`.
- **`forecast_ttm_univariate`/`forecast_ttm_mc_dropout` expect the input tensor moved to the model's device** — trained regime models live on CUDA, so build `x` with `.to(device)` or you get an `addmm` device mismatch.
- **MC-dropout band is over-confident** (first calibration: ~33% z=1 coverage vs 68.3% expected) — treat `forecast_std` as a lower bound on uncertainty, not a calibrated interval; `regime_calibrate.py` re-measures coverage.

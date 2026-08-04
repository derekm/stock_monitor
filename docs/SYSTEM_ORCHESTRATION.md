# System Orchestration

How the stock_monitor programs chain together: data in → analytics → dashboard. Read this before running anything in production, and before asking an agent to "run the analytics."

## 0. System architecture

```mermaid
flowchart TB
  subgraph SPINE["Data spine (DATA_DIR — single source of truth)"]
    DP[daily_prices.parquet]
    FUND[fundamentals.parquet]
    MS[monitored_stocks.parquet]
    HOLD[portfolio_holdings / trades]
    EXO[exogenous_panel.parquet]
    SP[sp500_constituents / sp500_changes]
    ALERT[alerts_config.parquet]
  end

  subgraph INGEST["Ingest (dedicated writer scripts)"]
    UP[update_prices / backfill_historical]
    UF[update_fundamentals / fundamentals_history]
    MG[manage_stocks / manage_alerts]
    PS[parse_sp500*]
  end

  subgraph LOOPS["Analytics loops (read spine → write outputs)"]
    SCR["Screen: preferred_metrics → inclusion_criteria → stress_dual_pass"]
    REG["Regime: hmm_regime_detection + kalman_state_estimates → regime_aware_constraints / factor_rotation_defense / rebalance_calendar"]
    CORR["Corr: allpairs / crisis / rolling / cross_asset"]
    RISK["Risk: portfolio_optimization / risk_parity / robust_cov / vol_target / kelly"]
    IDX["Indexes: build_* → fisher_index / run_fisher_duckdb"]
    FC["Forecast: ttm_features+ttm_exogenous → ttm_backfill → granite_daily → forecast_granite → analyze_granite_forecasts"]
  end

  subgraph SVC["Services (start_dashboard.sh)"]
    GS["granite_service :5055"]
    PSVC["pipeline_service :5056"]
    AS["analytics_service :8767"]
    WEB["static :8765"]
  end

  BROWSER["Dashboard / browser"]

  INGEST --> SPINE
  SPINE --> LOOPS
  LOOPS --> GS
  LOOPS --> AS
  AS --> PSVC
  PSVC -.subprocess.-> LOOPS
  GS --> WEB
  AS --> WEB
  WEB --> BROWSER
  RUN["run_daily_automation.py (master orchestrator)"] --> LOOPS
```

> Two equivalent entry points drive the analytics loops: `run_daily_automation.py`
> (CLI) and `analytics_service.py`'s `POST /run/all-daily` (dashboard). Both run
> the same ordered jobs; choose whichever fits the context.

## 1. The data spine (shared inputs)

Almost every program reads from a small set of canonical parquet/CSV tables in `DATA_DIR` (loaded via `data_access.py`). Change these and everything downstream moves:

| Table | Written by | Read by |
|---|---|---|
| `daily_prices.parquet` | `update_prices.py`, `backfill_historical.py` (yfinance) | Fisher indexes, TTM/forecasts, correlations, risk, backtests |
| `fundamentals.parquet` | `update_fundamentals.py`, `fundamentals_history.py` | screens, preferred_metrics, inclusion, dupont |
| `monitored_stocks.parquet` | `manage_stocks.py` | index builds, ticker resolution, screens |
| `portfolio_holdings.parquet` | external/Robinhood export | portfolio_report, risk, optimization |
| `trades.parquet` | external (Robinhood fills) | portfolio_report, perf vs benchmarks |
| `sector_prices.parquet` | `cross_asset_analysis.py save-sector-prices` | exogenous features, forecasting |
| `exogenous_panel.parquet` | `ttm_exogenous.py` | Granite TTM forecast channels (market/sector returns, dispersion) |
| `granite_series_cache.parquet` | `granite_daily.py` | cached series for the daily forecast loop |
| `sp500_constituents.parquet` | `parse_sp500.py` | S&P tracking (`sp_universe_tracking`, `sp_index_methodology`) |
| `sp500_changes.parquet` | `parse_sp500_changes.py` / `parse_tickerleague_changes.py` | S&P add/remove event log (`sp_index_methodology`, `sp_history_simulation`) |
| `alerts_config.parquet` | seed / `manage_alerts.py` | `check_alerts.py` |

> Schema details for every output above (and the ~120 others) live in [SCHEMAS.md](SCHEMAS.md).

## 2. The pipeline (a typical day)

```
                         ┌─────────────── ingest ───────────────┐
   yfinance ──► update_prices.py  /  backfill_historical.py ──► daily_prices.parquet
   manual   ──► update_fundamentals.py / fundamentals_history.py ──► fundamentals.parquet
   roster   ──► manage_stocks.py ──► monitored_stocks.parquet
                         └─────────────────────────────────────┘

              ┌────────────── analytics (maintain_analytics.py hub) ──────────────┐
   screens:   preferred_metrics → inclusion_criteria → stress_dual_pass
   indexes:   build_index / build_growth_tech_index / build_defensive_index → fisher_index (run_fisher_duckdb)
   risk:      portfolio_optimization / risk_parity_analytics / robust_covariance / vol_target / kelly
   regimes:   hmm_regime_detection → regime_correlation_breakdown / regime_aware_constraints / kalman_state_estimates
   corr:      allpairs_correlations / crisis_correlation / cross_asset_analysis / rolling_*
   forecasts: ttm_features + ttm_exogenous → granite_backfill/ttm_backfill (pretrain) → granite_daily → forecast_granite
              └────────────────────────────────────────────────────────────────────┘

              ┌────────────── publish ──────────────┐
   export_dashboard_data.py ──► dashboard_data/data.json  (+ build_data_catalog.py → data_catalog.json)
              └─────────────────────────────────────┘
```

`run_daily_automation.py` is the master orchestrator that runs the analytics steps in order (preferred → inclusion → stress → rolling → allpairs → fundamentals snapshot → dupont → growth-tech → export). Use it instead of calling steps by hand.

## 3. Services (started by `start_dashboard.sh`)

`start_dashboard.sh` launches **four** processes and a static file server, then blocks until Ctrl+C (which kills all four). Ports are env-overridable:

| Process | Port (env) | Role | Doc |
|---|---|---|---|
| `export_dashboard_data.py` (+ `build_data_catalog.py`) | — | writes `dashboard_data/data.json` + `data_catalog.json` at startup | [export_dashboard_data.md](export_dashboard_data.md) |
| `granite_service.py` | 5055 (`PORT_API`) | live Granite/fallback forecasts (HTTP) | [granite_service.md](granite_service.md) |
| `pipeline_service.py` | 5056 (`PORT_PIPE`) | job runner: rerun prices/backfill/analytics/export/forecast_bt/monte_carlo/all as subprocesses | [pipeline_service.md](pipeline_service.md) |
| `analytics_service.py` | 8767 (`PORT_ANALYTICS`; script default 8765) | dashboard ops hub: trigger daily jobs (`POST /run/all-daily`, `/run/update-prices`, `/run/preferred-metrics`, `/run/rolling`, `/run/export-dashboard`, …), read tables, data-integrity | [analytics_service.md](analytics_service.md) |
| `python -m http.server` | 8765 (`PORT_WEB`) | static dashboard (`index.html`) | — |

> The script default for `analytics_service` is **8765**; `start_dashboard.sh` overrides it to **8767** so the static server can keep 8765. If you launch `analytics_service.py` by hand, it listens on 8765 unless you pass `--port`.

After launch:
- Dashboard: http://127.0.0.1:8765/index.html
- Forecasts API: http://127.0.0.1:5055/health
- Pipeline jobs: http://127.0.0.1:5056/jobs
- Analytics API: http://127.0.0.1:8767/health

## 4. Forecast subsystem (Granite TTM)

Forecasts are the most stateful part. Order matters:

1. **Build panels**: `ttm_features.py` (multivariate panels from `daily_prices`) + `ttm_exogenous.py` (market/sector/dispersion channels → `exogenous_panel.parquet`).
2. **Pre-train** history once (or when data grows): `ttm_backfill.py` / `granite_backfill.py` (thin shim) / `train_adjusted_full.py` (adj-close config) produce global adjusted checkpoints under `checkpoints/`. `window_padding.py` pads sub-512-day tickers with a rescaled market proxy so the fixed 512-token TTM context is valid.
3. **Daily**: `granite_daily.py` runs the 512→96-day model with continual retraining on prior-day actuals (caching series in `granite_series_cache.parquet`); `forecast_granite.py` produces `forecasts_granite.csv/.parquet` used by `granite_service.py`.
4. **Score**: `analyze_granite_forecasts.py` backtests the forecasts (writes `forecast_backtest_metrics.csv/.parquet`); `forecast_reliability.py` ranks setups on the actual holdings; `research_hygiene.py` reports reliability.
5. **Anomalies**: `tspulse_anomaly.py` scans for outliers.

Checkpoints live under `checkpoints/` (Git-ignored / large). Never delete them mid-run.

## 5. S&P tracking subsystem

Independent reimplementation of S&P 500 inclusion/exclusion, scored against actuals:

`parse_sp500.py` → `sp500_constituents.parquet` → `parse_sp500_changes.py` / `parse_tickerleague_changes.py` (event logs) → `sp_index_methodology.py` (reimplementation + tiers) → `sp_universe_tracking.py` (503-constituent tracking) → `reconcile_sp500.py` (fix `monitored_stocks`).

## 6. What an agent should know before "running analytics"

- Always start from `run_daily_automation.py` (or the dashboard's `analytics_service` → `/run/all-daily`), not individual scripts.
- Data must be fresh: if `daily_prices.parquet` is stale, run `update_prices.py --fetch --days N` first.
- The dashboard reads `dashboard_data/data.json`; if tables look empty, re-run `export_dashboard_data.py`.
- Don't hand-edit the base parquet tables; use the dedicated writer scripts (`manage_stocks`, `update_fundamentals`, `update_prices`).
- Forecasts need pretrained checkpoints; if `forecasts_granite.parquet` is missing, run `granite_backfill.py`/`ttm_backfill.py` then `granite_daily.py`.
- Full program catalog with cross-links: see each `docs/<script>.md` and [SCHEMAS.md](SCHEMAS.md).

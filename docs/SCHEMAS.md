# Output Schemas (catalog)

This document is the **single source of truth for output file schemas** produced by the stock_monitor programs. Rather than repeat column definitions in every program doc, each output below is assigned a **schema family**; program docs link here. Column lists under each family are *representative of that family's shape* — open the producing script for the exact DataFrame built.


## Schema families

Each output belongs to exactly one **family** (a shape + a *job* in the stack). The
families are ordered by how they compose: **base tables → screens → regime →
risk/weights → forecasts → indexes → correlation structure → summaries**, with
auxiliary tables feeding several stages.

### Base parquet table  (`base_table`)

- The canonical, slowly-changing inputs everything else reads. Shape varies
  (`daily_prices`: date,ticker,OHLCV,adj_close; `fundamentals`: ticker,as_of_date,
  valuation ratios; `monitored_stocks`: ticker,sector,index membership;
  `portfolio_holdings`/`trades`: the real book; `exogenous_panel`: per-date
  market/sector/dispersion channels; S&P tables: `sp500_constituents`,
  `sp500_changes`). **Never hand-edit these** — use the dedicated writers
  (`update_prices`, `update_fundamentals`, `manage_stocks`, `manage_alerts`,
  `parse_sp500*`, `backfill_*`). They are the single source of truth.

### `fundamentals.parquet` column naming (canonical since 2026-08)

Column names state the **period basis** explicitly; the old bare-vs-`ttm_` prefix
did not reliably encode it.

| Basis | Suffix | Columns |
| --- | --- | --- |
| One fiscal quarter | `_quarterly` | `revenue_quarterly`, `net_income_quarterly`, `operating_income_quarterly` |
| Trailing twelve months | `_ttm` | `revenue_ttm`, `net_income_ttm`, `operating_income_ttm`, `operating_cash_flow_ttm`, `capital_expenditure_ttm` |
| Point-in-time balance | *(none)* | `total_assets`, `total_debt`, `total_liabilities`, `shareholders_equity`, `cash_and_equivalents`, `shares_outstanding` |

A `_ttm` column is ~4x its `_quarterly` partner (measured median 3.67–3.68).
`test_basic.test_fundamentals_canonical_schema` fails if that ratio degenerates
toward 1.0, which is how a twelve-month sum previously ended up under a quarterly
name.

Renamed away — **do not reintroduce**: `equity`, `stockholders_equity` →
`shareholders_equity`; `assets` → `total_assets`; `cash` →
`cash_and_equivalents`; `total_revenue`, `revenue` → `revenue_quarterly`;
`net_income` → `net_income_quarterly`; `ttm_*` → `*_ttm`. The bare
`operating_cash_flow` and `capital_expenditure` already held TTM values (ratio
1.000 against their `ttm_` partners) and map to `*_ttm`, **not** `*_quarterly`.

`shares` was **dropped**, not renamed: it disagreed with `shares_outstanding` on
3,547 of 4,866 shared rows (FITB 2010-09-30 held 7.96e14 — 796 trillion shares),
and 45 of its 73 unique rows were 0.0.

`total_debt` and `total_liabilities` are **distinct** (median ratio 2.515 — debt
is a subset). A writer bug used to copy one into the other.

Known pre-existing defect: 211 rows have `shares_outstanding > 1e11` (PCG
preferred series at 5.14e14, all `source=edgar`). The current writer rejects
values ≥1e11, but the legacy rows remain.

Migration is re-runnable and idempotent: `python migrate_fundamentals_schema.py
--dry-run` then `--apply`.


### Screen / decision  (`screen_decision`)

- Decision tables keyed by `ticker`: boolean/label columns for each gate leg
  (`roe`,`roic`,`debt_to_equity`,`ev_ebitda`,`pb_ratio`,`mktcap_to_assets`), a
  `decision`/`action` label (INCLUDE_CORE / VALUE / QUALITY / SATELLITE / WATCH /
  AVOID), and `fail_legs`. This is the **policy layer** — the canonical dual-screen
  gate lives in `quality_gate_bridge` (mirroring the `stockmagic` library) and is
  consumed by `inclusion_criteria`, `preferred_metrics`, `threshold_logic`. Outputs
  here are *policy decisions*, not measurements.

### Regime / state table  (`regime_state`)

- `date`, a `regime`/`state` label (Calm/Stress/High-vol), and state probabilities
  or latent estimates (`p_calm`,`p_stress`, `mkt`, `log_vol`, `kalman_state`,
  `entropy`). **This is the master risk switch**: regime state drives
  `regime_aware_constraints` (which caps relax), `factor_rotation_defense` (which
  sleeve is overweight), `rebalance_calendar`, and feeds `monte_carlo` /
  `mcmc_regimes`. Three estimators triangulate it — HMM (`hmm_regime_detection`),
  Kalman (`kalman_state_estimates`), and VAR — plus `vix_term_structure` as an
  offline vol-slope proxy.

### Weights / performance  (`weights_performance`)

- Strategy/name-level: `ticker`/`strategy`, `weight` (fraction), risk/return stats
  (`ret`,`vol`,`sharpe`,`max_dd`) and per-name `rc` (risk contribution). The
  **allocation layer** — takes screen + regime + risk inputs and produces target
  weights (ERC/GMV/inv-vol in `portfolio_optimization`, `risk_parity_analytics`;
  factor sleeves in `factor_rotation_defense`; hedges in `tail_risk_hedging`).

### Forecast / anomaly  (`forecast_anomaly`)

- Forecast rows: `ticker`, `as_of`/`forecast_date`, `horizon`, `pct_change` (+ `close`/
  `history` for charts); anomaly rows: `ticker`, `date`, `z_*` scores, `flag`. The
  **Granite TTM subsystem**: `ttm_features`+`ttm_exogenous` build panels →
  `ttm_backfill`/`train_adjusted_full` pretrain (adj-close) → `granite_daily`
  continual-retrains → `forecast_granite` emits these → `analyze_granite_forecasts`
  scores them → `granite_service` serves them. `tspulse_anomaly` flags bad prints.
  **LLM desk notes** (`forecast_llm.parquet`): `date`, `ticker`, `profile`
  (`value` | `exuberant` | `compounder`), `forecast_dir`, `forecast_prob`, `horizon_days`,
  `narrative`, `damodaran_narrative`. Long table; resume on `(date, ticker, profile)`.
  Multi-date via `--dates-file` / `--as-of`: latest date keeps full snapshot panels;
  historical dates are PIT-only (undated snapshot panels dropped to avoid lookahead).
  `conformal_bands.parquet`: `date`, `regime`, `lower`, `upper`, `horizon_days`, `n`,
  `coverage_rate`, `coverage_flag` — split-conformal coverage of LLM prob vs forward
  market return over the row's own horizon (bar 0.90).
  Press sidecar: article text in `news_articles/YYYY/MM/DD/{id}.md`; mentions in
  `ticker_news_mentions.parquet`.

### Index level series  (`index_levels`)

- Time series: `date`, `level` (index value, base 100), component `return` columns.
  **Quantity-weighted truth** vs the cap-weighted S&P: `fisher_index` /
  `run_fisher_duckdb` (DuckDB is system-of-record) chain Laspeyres·Paasche·Fisher
  from close (price) × volume (quantity); `build_index` / `build_defensive_index` /
  `build_growth_tech_index` assemble the sleeves; `live_index_backtest` Sharpe-tests
  them.

### Bogle fund series  (`bogle_fund`)

- `bogle_{tmi,qmi,qmi_strict,bpi,pmi}.parquet` — `date`, `level` (base 1000),
  `ret_gross`, `ret_net`, `expense_drag`, `turnover_cost`, `turnover`, `fund`,
  `weight_method`, `rebalance_freq`, `expense_bps`, `turnover_bps`, plus the
  Fisher arm (`fisher_p`, `fisher_q`, `fisher_p_net`, `nominal_sqrt_fisher`).
  `weight_method` is `cap_weighted` (TMI), `equal_weighted` (QMI/QMI_STRICT/BPI)
  or `equal_weighted_capped` (PMI, 5% single-name cap).
- `bogle_*_turnover.parquet` — `date`, `turnover` (one-way), `turnover_cost`, `fund`;
  rebalance rows only.
- **Universe split:** TMI/QMI/BPI gate `exchange ∈ {NMS,NYQ,NCM,NGM,ASE}`;
  **PMI gates the complement** (`exchange_mode="exclude"`), so `TMI ∪ PMI` is the
  complete market and `TMI ∩ PMI = ∅`. Gates documented in `docs/bogle_inclusion.md`.

### Regime clustering  (`regime_clustering`)

- `regime_clusters.parquet` — `ticker`, `cluster` (int), `hrp_order` (HRP
  quasi-diagonal seriation index), `sector` (GICS baseline), `metric`
  (`corr`|`dcor`), `linkage`, `k`, `cluster_name` (named by dominant sector,
  e.g. `technology_77`, `energy_82`, `financial_services_100`), `peer_group`
  (hybrid: cluster where tighter than GICS, else falls back to GICS sector).
  Covers ~4,670 liquid listed stocks (61% of the 7,627-name universe; remaining
  are ETFs/warrants/preferreds with no GICS sector).
- **Clustered sectors as a better GICS.** `peer_group` is the drop-in
  replacement for `sector` in `peer_analytics` / `cross_section`: it uses the
  cluster name where the cluster is tighter than GICS (e.g.
  `financial_services_76`, `mixed_healthcare`), and falls back to the GICS
  sector where the cluster is looser (e.g. `basic_materials_89` → `Basic
  Materials`).
- `regime_cluster_dispersion.parquet` — one row per grouping
  (`gics_sector`, `hrp_cluster`, `reduction_pct`) with `dispersion`
  (size-weighted std of within-group pairwise corr), `n_groups`, `n_pairs`,
  `mean_within_corr`, `singletons`, `bar_pct` (20), `passes`, `n_assets`, `years`.
- `regime_cluster_sweep.parquet` — `metric`, `linkage`, `years`,
  `reduction_pct`, `passes`. **Read the sweep, not one cell**: the ≥20% bar
  outcome is linkage-dependent (5/8 configs pass; +7.3% to +28.1%).
- **Not** a regime *date* table: `hmm_regime_states` (family `regime_state`) is
  the date→regime series. This family groups ASSETS.

### Correlation matrix  (`correlation_matrix`)

- `index` (row label, ticker or sector) × one column per entity, pairwise
  coefficient in [-1,1]. Square or long-form `a,b,corr`. **Encodes the
  diversification-fails-in-crisis fact**: calm pairwise ~0.15, crisis ~0.45+, sector
  crisis higher — which is *why* regime switching, hedges, and cash buffers exist.
  Produced at many horizons (rolling 21/63/126, ALLPAIRS history, crisis vs calm,
  regime-conditioned) and by `hmm_regime_detection` (transition matrix).

### Summary / metrics  (`summary_metrics`)

- One-row/few-row aggregates: `_summary`/`_stats`/`_metrics` with scalars (counts,
  vol, sharpe, avg corr, pass counts, reliability ranks), or long-form `name,value`.
  The dashboards and `research_hygiene` / `forecast_reliability` consume these.

### Lookthrough / pro forma  (`lookthrough`)

Pro forma combined financials during acquisition windows. Two provenance columns:
`data_provenance` (standalone | lookthrough_proforma) and `lookthrough_source`
(comma-separated list of combined tickers, or None for standalone). Produced by
`lookthrough_engine.py` from `fundamentals.parquet` + `corporate_actions.parquet`.

### Corporate actions  (`corporate_actions`)

Acquisition, merger, and delisting records. Tracks acquirer/target tickers,
announcement and completion dates, purchase terms, and lookthrough window
boundaries. Produced by `acquisition_backfill.py` and consumed by
`lookthrough_engine.py`.

### Fiscal year map  (`fiscal_year_map`)

Per-ticker fiscal year end month (1-12) detected from equity series dates.
Used to align calendar-quarter EDGAR frames to the filer's fiscal calendar.
Produced by `edgar_companyfacts_v2.py` and `acquisition_backfill.py`.

### Expired price history  (`expired_prices`)

Preserved daily price history for delisted/acquired tickers. Kept out of the
main `daily_prices/` to keep the active universe clean, but retained
for backtesting and analytics. Produced by `data_validation.py` cleanup.

- Supporting tables: `sector_tickers` (ticker,sector,SECT_* slug), `vix_term_structure`
  (tenor, iv), membership/catalog listings. Feed the screen layer and the
  forecasting exogenous channels.

### Other  (`other`)

- See producing script (mixed-shape outputs that don't fit a family cleanly).

### Pair engine  (`pair_engine`)

- `pair_id` (A|B), `group` (industry/sector), cointegration stats
  (`coint_t`,`p_value`,`beta`,`half_life`), selection flags
  (`fdr_survive`,`usable`,`fold`), live `z_now`; trades carry `entry_date`,
  `exit_date` (DATE), `entry_z`/`exit_z`, `bars_held`, `exit_reason`
  (revert/stop/time), `hedged_pnl`, `z_pnl`. Produced only by `pair_engine.py`;
  all stats are walk-forward OOS.

### Earnings catalyst  (`earnings`)

- Per-ticker earnings signal rows: `ticker`, `next_earnings_date` (DATE),
  `surprise_pct`, `pre_mom_pctile`/`pre_mom_flag`, `iv_vs_realized`/`iv_rich`,
  `expected_drift_20d`, `catalyst_score`; plus drift-bucket aggregates
  (`bucket`, `n_events`, `drift_5d/20d/63d`). Produced by `earnings_catalyst.py`
  from `earnings_calendar.parquet` (`update_earnings.py`).

### Cross-section  (`cross_section`)

- `rebalance_date` (DATE), `ticker`, `bucket` (1=short … 5=long) rankings;
  daily `long`/`short`/`long_short`/`equal_weight_long` returns; OOS stats vs
  baseline incl. `sector_exposure_abs_dev_avg`. Produced only by
  `cross_section.py`; factors are point-in-time (as-of fundamentals +
  trailing momentum).

### Signal aggregate  (`aggregate`)

- `ticker`, per-family normalized scores (`preferred`,`peer`,`cross`,`pair`,
  `earnings`), `composite`, `rank`; plus per-family `ic`, `n`, `weight`,
  `weight_norm`. Produced only by `signal_aggregator.py`; weights are
  trailing-window OOS (IC measured at cutoff − 21d).


## Full output catalog

| Output file | Producing script | Family |
|---|---|---|
| `allpairs_asset_corr_history.csv` | `allpairs_correlations.py` | Correlation matrix |
| `allpairs_asset_corr_latest.csv` | `allpairs_correlations.py` | Correlation matrix |
| `allpairs_corr_summary.csv` | `allpairs_correlations.py` | Correlation matrix |
| `allpairs_sector_corr_history.csv` | `allpairs_correlations.py` | Correlation matrix |
| `allpairs_sector_corr_latest.csv` | `allpairs_correlations.py` | Correlation matrix |
| `asset_correlation_matrix.csv` | `inclusion_criteria.py` | Correlation matrix |
| `asset_sector_correlations.csv` | `cross_asset_analysis.py` | Correlation matrix |
| `correlation_stability_metrics.csv` | `maintain_analytics.py` | Correlation matrix |
| `crisis_avg_corr_timeseries.csv` | `crisis_correlation.py` | Correlation matrix |
| `crisis_correlation_pairs.csv` | `crisis_correlation.py` | Correlation matrix |
| `crisis_correlation_summary.csv` | `crisis_correlation.py` | Correlation matrix |
| `fertilizer_correlation_matrix.csv` | `maintain_analytics.py` | Correlation matrix |
| `growth_tech_corr_stability.csv` | `growth_tech_analytics.py` | Correlation matrix |
| `growth_tech_correlation_matrix.csv` | `growth_tech_analytics.py` | Correlation matrix |
| `growth_tech_rolling_corr.csv` | `growth_tech_analytics.py` | Correlation matrix |
| `growth_tech_sleeve_correlations.csv` | `growth_tech_analytics.py` | Correlation matrix |
| `hmm_2state_regime_correlations.csv` | `maintain_analytics.py` | Correlation matrix |
| `hmm_transition_matrix.csv` | `hmm_regime_detection.py` | Correlation matrix |
| `hmm_transition_matrix.csv` | `monte_carlo.py` | Correlation matrix |
| `kalman_correlations.csv` | `maintain_analytics.py` | Correlation matrix |
| `kalman_gain_path.csv` | `kalman_gain_analysis.py` | Regime / state table |
| `kalman_gain_summary.csv` | `kalman_gain_analysis.py` | Regime / state table |
| `kalman_state_estimates.csv` | `kalman_state_estimates.py` | Regime / state table |
| `kalman_state_summary.csv` | `kalman_state_estimates.py` | Regime / state table |
| `mcmc_regime_means.csv` | `mcmc_regimes.py` | Regime / state table |
| `mcmc_transition_draws.csv` | `mcmc_regimes.py` | Regime / state table |
| `mcmc_regime_summary.csv` | `mcmc_regimes.py` | Regime / state table |
| `regime_corr_breakdown.csv` | `regime_correlation_breakdown.py` | Correlation matrix |
| `regime_corr_pair_delta.csv` | `regime_correlation_breakdown.py` | Correlation matrix |
| `regime_sector_corr.csv` | `regime_correlation_breakdown.py` | Correlation matrix |
| `rolling_corr_avg_timeseries.csv` | `rolling_correlation_windows.py` | Correlation matrix |
| `rolling_corr_stability_by_asset.csv` | `rolling_correlation_windows.py` | Correlation matrix |
| `rolling_cross_asset_correlations.csv` | `cross_asset_analysis.py` | Correlation matrix |
| `rolling_sector_corr_windows.csv` | `rolling_correlation_windows.py` | Correlation matrix |
| `rolling_sector_correlations.csv` | `maintain_analytics.py` | Correlation matrix |
| `sector_correlation_matrix.csv` | `cross_asset_analysis.py` | Correlation matrix |
| `sector_correlation_matrix.csv` | `maintain_analytics.py` | Correlation matrix |
| `sector_correlation_matrix_latest.csv` | `inclusion_criteria.py` | Correlation matrix |
| `hmm_2state_regimes.csv` | `maintain_analytics.py` | Regime / state table |
| `hmm_posterior_analysis.csv` | `hmm_posterior_analysis.py` | Regime / state table |
| `hmm_posterior_analysis.csv` | `posterior_entropy_dynamics.py` | Regime / state table |
| `hmm_posterior_summary.csv` | `hmm_posterior_analysis.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `hmm_posterior_analysis.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `hmm_regime_detection.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `kalman_gain_analysis.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `kalman_state_estimates.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `monte_carlo.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `posterior_entropy_dynamics.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `regime_aware_constraints.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `regime_correlation_breakdown.py` | Regime / state table |
|| `hmm_regime_states.parquet` | `threshold_logic.py` | Regime / state table |
| `hmm_regime_summary.csv` | `hmm_regime_detection.py` | Regime / state table |
| `hmm_regimes.csv` | `black_litterman_views.py` | Regime / state table |
| `hmm_regimes.csv` | `buy_candidates.py` | Regime / state table |
| `hmm_regimes.csv` | `rebalance_calendar.py` | Regime / state table |
| `hmm_transition_triggers.csv` | `regime_aware_constraints.py` | Regime / state table |
| `hmm_uncertain_days.csv` | `hmm_posterior_analysis.py` | Regime / state table |
| `kalman_gain_path.csv` | `kalman_gain_analysis.py` | Regime / state table |
| `kalman_gain_summary.csv` | `kalman_gain_analysis.py` | Regime / state table |
| `kalman_state_estimates.csv` | `kalman_state_estimates.py` | Regime / state table |
| `kalman_state_summary.csv` | `kalman_state_estimates.py` | Regime / state table |
| `mcmc_regime_means.csv` | `mcmc_regimes.py` | Regime / state table |
| `mcmc_regime_summary.csv` | `mcmc_regimes.py` | Regime / state table |
| `mcmc_transition_draws.csv` | `mcmc_regimes.py` | Regime / state table |
| `posterior_entropy_dynamics.csv` | `posterior_entropy_dynamics.py` | Regime / state table |
| `posterior_entropy_summary.csv` | `posterior_entropy_dynamics.py` | Regime / state table |
| `regime_aware_dual_pass.csv` | `regime_aware_constraints.py` | Regime / state table |
| `regime_aware_summary.csv` | `regime_aware_constraints.py` | Regime / state table |
| `regime_constraint_binding.csv` | `regime_aware_constraints.py` | Regime / state table |
| `black_litterman_weights.csv` | `black_litterman.py` | Weights / performance |
| `black_litterman_weights_from_views.csv` | `black_litterman_views.py` | Weights / performance |
| `factor_rotation_performance.csv` | `factor_rotation_defense.py` | Weights / performance |
| `factor_rotation_weights.csv` | `factor_rotation_defense.py` | Weights / performance |
| `factor_sleeve_returns.csv` | `factor_rotation_defense.py` | Weights / performance |
| `factor_groups.csv` | `factor_rotation_defense.py` | Groups (catalog: group → type) |
| `factor_group_members.csv` | `factor_rotation_defense.py` | Groups (join: group, ticker, valid_from, valid_to) |
| `growth_tech_sleeve_performance.csv` | `growth_tech_analytics.py` | Weights / performance |
| `tail_risk_hedge_performance.csv` | `tail_risk_hedging.py` | Weights / performance |
| `anomalies_tspulse.csv` | `tspulse_anomaly.py` | Forecast / anomaly |
| `forecast_backtest_metrics.csv` | `analyze_granite_forecasts.py` | Forecast / anomaly |
| `forecast_backtest_metrics.csv` | `forecast_granite.py` | Forecast / anomaly |
| `forecast_backtest_metrics.csv` | `forecast_reliability.py` | Forecast / anomaly |
| `forecast_backtest_metrics.csv` | `research_hygiene.py` | Forecast / anomaly |
| `forecast_reliability_detail.csv` | `forecast_reliability.py` | Forecast / anomaly |
| `forecast_reliability_rank.csv` | `forecast_reliability.py` | Forecast / anomaly |
| `forecast_reliability_report.csv` | `research_hygiene.py` | Forecast / anomaly |
| `forecasts_granite.csv` | `analyze_granite_forecasts.py` | Forecast / anomaly |
| `forecasts_granite.csv` | `forecast_granite.py` | Forecast / anomaly |
| `forecasts_granite.parquet` | `analyze_granite_forecasts.py` | Forecast / anomaly |
| `forecasts_granite.parquet` | `forecast_granite.py` | Forecast / anomaly |
| `forecasts_granite.parquet` | `granite_daily.py` | Forecast / anomaly |
| `forecast_llm.parquet` | `forecast_llm.py` | Forecast / anomaly |
| `binding_basket_risk.csv` | `binding_constraints_analysis.py` | Screen / decision |
| `binding_constraints_impact.csv` | `binding_constraints_analysis.py` | Screen / decision |
| `binding_near_miss_detail.csv` | `binding_constraints_analysis.py` | Screen / decision |
| `buy_candidates.csv` | `buy_candidates.py` | Screen / decision |
| `buy_candidates_top.csv` | `buy_candidates.py` | Screen / decision |
| `defensive_value_exploration.csv` | `inclusion_criteria.py` | Screen / decision |
| `dual_pass_sensitivity.csv` | `stress_dual_pass.py` | Screen / decision |
| `dual_pass_stress.csv` | `stress_dual_pass.py` | Screen / decision |
| `dual_screen_external_candidates.csv` | `dual_screen_analysis.py` | Screen / decision |
| `dual_screen_gap.csv` | `dual_screen_analysis.py` | Screen / decision |
| `exclusion_candidates.csv` | `inclusion_criteria.py` | Screen / decision |
| `factor_panel.csv` | `buy_candidates.py` | Screen / decision |
| `factor_panel.csv` | `factor_panel.py` | Screen / decision |
| `factor_panel_top.csv` | `factor_panel.py` | Screen / decision |
| `growth_tech_backtest_stats.csv` | `growth_tech_analytics.py` | Screen / decision |
| `inclusion_candidates.csv` | `inclusion_criteria.py` | Screen / decision |
| `inclusion_walkforward.csv` | `research_hygiene.py` | Screen / decision |
| `index_backtest_stats.csv` | `live_index_backtest.py` | Screen / decision |
| `index_backtest_stats.csv` | `maintain_analytics.py` | Screen / decision |
|| `momentum_ic.csv` | `momentum_analytics.py` | Screen / decision ||
|| `momentum_metrics.parquet` | `buy_candidates.py` | Screen / decision ||
|| `momentum_metrics.parquet` | `factor_panel.py` | Screen / decision ||
|| `momentum_metrics.parquet` | `momentum_analytics.py` | Screen / decision ||
|| `momentum_quintiles.parquet` | `momentum_analytics.py` | Screen / decision ||
|| `momentum_research_backtest.parquet` | `momentum_research_backtest.py` | Screen / decision ||
|| `momentum_research_confidence.parquet` | `momentum_research_confidence.py` | Screen / decision ||
|| `fractal_windows_backtest.parquet` | `fractal_windows_backtest.py` | Screen / decision ||
|| `fractal_windows_backtest.parquet` | `fractal_windows_backtest_gpu.py` | Screen / decision ||
|| `near_dual_candidates.csv` | `inclusion_criteria.py` | Screen / decision ||
| `portfolio_risk_summary.csv` | `risk_metrics_ext.py` | Screen / decision |
| `preferred_screen_hits.csv` | `preferred_metrics.py` | Screen / decision |
| `risk_metrics_ext.csv` | `buy_candidates.py` | Screen / decision |
| `risk_metrics_ext.csv` | `risk_metrics_ext.py` | Screen / decision |
| `rolling_screen_stability.csv` | `rolling_window_analysis.py` | Screen / decision |
| `screen_backtest.csv` | `fundamentals_history.py` | Screen / decision |
| `threshold_logic_screen.csv` | `threshold_logic.py` | Screen / decision |
| `defensive_value_index.parquet` | `build_defensive_index.py` | Index level series |
| `fertilizer_index.parquet` | `build_index.py` | Index level series |
| `fisher_indexes.parquet` | `fisher_index.py` | Index level series |
| `fisher_indexes_duckdb.parquet` | `run_fisher_duckdb.py` | Index level series |
| `growth_tech_index.parquet` | `build_growth_tech_index.py` | Index level series |
| `growth_tech_index_levels.parquet` | `build_growth_tech_index.py` | Index level series |
| `growth_tech_index_levels.parquet` | `growth_tech_analytics.py` | Index level series |
| `growth_tech_index_levels_compare.csv` | `growth_tech_analytics.py` | Index level series |
| `growth_tech_index_levels_compare.parquet` | `growth_tech_analytics.py` | Index level series |
| `index_levels_1y.csv` | `live_index_backtest.py` | Index level series |
| `index_levels_1y.parquet` | `live_index_backtest.py` | Index level series |
| `index_levels_1y.parquet` | `maintain_analytics.py` | Index level series |
| `alerts_config.parquet` | `check_alerts.py` | Base parquet table |
| `alerts_config.parquet` | `manage_alerts.py` | Base parquet table |
| `fundamentals.parquet` | `backfill_constituents.py` | Base parquet table |
| `fundamentals.parquet` | `binding_constraints_analysis.py` | Base parquet table |
| `fundamentals.parquet` | `build_defensive_index.py` | Base parquet table |
| `fundamentals.parquet` | `build_growth_tech_index.py` | Base parquet table |
| `fundamentals.parquet` | `data_access.py` | Base parquet table |
| `fundamentals.parquet` | `data_integrity.py` | Base parquet table |
| `fundamentals.parquet` | `data_integrity_deep.py` | Base parquet table |
| `fundamentals.parquet` | `dual_screen_analysis.py` | Base parquet table |
| `fundamentals.parquet` | `dupont_analysis.py` | Base parquet table |
| `fundamentals.parquet` | `factor_rotation_defense.py` | Base parquet table |
| `fundamentals.parquet` | `fundamentals_history.py` | Base parquet table |
| `fundamentals.parquet` | `inclusion_criteria.py` | Base parquet table |
| `fundamentals.parquet` | `preferred_metrics.py` | Base parquet table |
| `fundamentals.parquet` | `regime_aware_constraints.py` | Base parquet table |
| `fundamentals.parquet` | `stress_dual_pass.py` | Base parquet table |
| `fundamentals.parquet` | `tail_risk_hedging.py` | Base parquet table |
| `fundamentals.parquet` | `threshold_logic.py` | Base parquet table |
| `fundamentals.parquet` | `update_fundamentals.py` | Base parquet table |
| `fundamentals.parquet` | `backfill_edgar.py` | Base parquet table |
| `fundamentals_history.parquet` | `data_integrity.py` | Base parquet table |
| `fundamentals_pit.parquet` | `data_integrity.py` | Base parquet table |
| `fundamentals_yfinance.parquet` | `backfill_constituents.py` | Base parquet table |
| `granite_series_cache.parquet` | `granite_daily.py` | Base parquet table |
| `monitored_stocks.parquet` | `allpairs_correlations.py` | Base parquet table |
| `monitored_stocks.parquet` | `backfill_historical.py` | Base parquet table |
| `monitored_stocks.parquet` | `build_defensive_index.py` | Base parquet table |
| `monitored_stocks.parquet` | `build_growth_tech_index.py` | Base parquet table |
| `monitored_stocks.parquet` | `build_index.py` | Base parquet table |
| `monitored_stocks.parquet` | `check_alerts.py` | Base parquet table |
| `monitored_stocks.parquet` | `crisis_correlation.py` | Base parquet table |
| `monitored_stocks.parquet` | `cross_asset_analysis.py` | Base parquet table |
| `monitored_stocks.parquet` | `data_access.py` | Base parquet table |
| `monitored_stocks.parquet` | `data_integrity_deep.py` | Base parquet table |
| `monitored_stocks.parquet` | `dual_screen_analysis.py` | Base parquet table |
| `monitored_stocks.parquet` | `factor_rotation_defense.py` | Base parquet table |
| `monitored_stocks.parquet` | `fisher_index.py` | Base parquet table |
| `monitored_stocks.parquet` | `fisher_sector_baskets.py` | Base parquet table |
| `monitored_stocks.parquet` | `forecast_granite.py` | Base parquet table |
| `monitored_stocks.parquet` | `growth_tech_analytics.py` | Base parquet table |
| `monitored_stocks.parquet` | `inclusion_criteria.py` | Base parquet table |
| `monitored_stocks.parquet` | `index_registry.py` | Base parquet table |
| `monitored_stocks.parquet` | `maintain_analytics.py` | Base parquet table |
| `monitored_stocks.parquet` | `manage_stocks.py` | Base parquet table |
| `monitored_stocks.parquet` | `portfolio_optimization.py` | Base parquet table |
| `monitored_stocks.parquet` | `portfolio_report.py` | Base parquet table |
| `monitored_stocks.parquet` | `preferred_metrics.py` | Base parquet table |
| `monitored_stocks.parquet` | `regime_aware_constraints.py` | Base parquet table |
| `monitored_stocks.parquet` | `regime_correlation_breakdown.py` | Base parquet table |
| `monitored_stocks.parquet` | `risk_parity_analytics.py` | Base parquet table |
| `monitored_stocks.parquet` | `robust_covariance.py` | Base parquet table |
| `monitored_stocks.parquet` | `rolling_correlation_windows.py` | Base parquet table |
| `monitored_stocks.parquet` | `rolling_window_analysis.py` | Base parquet table |
| `monitored_stocks.parquet` | `run_fisher_duckdb.py` | Base parquet table |
| `monitored_stocks.parquet` | `tail_risk_hedging.py` | Base parquet table |
| `monitored_stocks.parquet` | `tspulse_anomaly.py` | Base parquet table |
| `monitored_stocks.parquet` | `ttm_exogenous.py` | Base parquet table |
| `monitored_stocks.parquet` | `ttm_features.py` | Base parquet table |
| `monitored_stocks.parquet` | `update_fundamentals.py` | Base parquet table |
| `monitored_stocks.parquet` | `update_prices.py` | Base parquet table |
| `monitored_stocks.parquet` | `vol_target.py` | Base parquet table |
| `daily_prices/` (OHLCV backfill) | `backfill_ohlcv.py` | Data maintenance. Backfills full open/high/low/close/volume history for the whole universe via yfinance; fills the OHLC gap in the close+volume-only history; preserves market_cap; resume-safe |
| `portfolio_holdings.parquet` | `data_access.py` | Base parquet table |
| `portfolio_holdings.parquet` | `growth_tech_analytics.py` | Base parquet table |
| `portfolio_holdings.parquet` | `inclusion_criteria.py` | Base parquet table |
| `portfolio_holdings.parquet` | `index_registry.py` | Base parquet table |
| `portfolio_holdings.parquet` | `maintain_analytics.py` | Base parquet table |
| `portfolio_holdings.parquet` | `portfolio_optimization.py` | Base parquet table |
| `portfolio_holdings.parquet` | `portfolio_report.py` | Base parquet table |
| `portfolio_holdings.parquet` | `preferred_metrics.py` | Base parquet table |
| `portfolio_holdings.parquet` | `risk_metrics_ext.py` | Base parquet table |
| `portfolio_holdings.parquet` | `risk_parity_analytics.py` | Base parquet table |
| `portfolio_holdings.parquet` | `robust_covariance.py` | Base parquet table |
| `portfolio_holdings.parquet` | `rolling_window_analysis.py` | Base parquet table |
| `portfolio_holdings.parquet` | `vol_target.py` | Base parquet table |
| `sp500_constituents.parquet` | `backfill_constituents.py` | Base parquet table |
| `sp500_constituents.parquet` | `parse_sp500.py` | Base parquet table |
| `sp500_changes.parquet` | `parse_sp500_changes.py` | Base parquet table |
| `sp500_changes.parquet` | `parse_tickerleague_changes.py` | Base parquet table |
| `sp500_changes_tickerleague.parquet` | `parse_tickerleague_changes.py` | Base parquet table |
| `sp500_universe_tracking.parquet` | `sp_universe_tracking.py` | Base parquet table |
| `exogenous_panel.parquet` | `tspulse_anomaly.py` | Base parquet table |
| `exogenous_panel.parquet` | `ttm_exogenous.py` | Base parquet table |
| `granite_series_cache.parquet` | `granite_daily.py` | Base parquet table |
| `trades.parquet` | `data_access.py` | Base parquet table |
| `trades.parquet` | `index_registry.py` | Base parquet table |
| `cross_asset_stability.csv` | `cross_asset_analysis.py` | Summary / metrics |
| `erc_gmv_summary.csv` | `portfolio_optimization.py` | Summary / metrics |
| `monte_carlo_path_stats.csv` | `monte_carlo.py` | Summary / metrics |
| `monte_carlo_summary.csv` | `monte_carlo.py` | Summary / metrics |
| `preferred_metrics.csv` | `black_litterman_views.py` | Summary / metrics |
| `preferred_metrics.csv` | `buy_candidates.py` | Summary / metrics |
| `preferred_metrics.csv` | `data_integrity_deep.py` | Summary / metrics |
| `preferred_metrics.csv` | `factor_panel.py` | Summary / metrics |
| `preferred_metrics.csv` | `factor_rotation_defense.py` | Summary / metrics |
| `preferred_metrics.csv` | `inclusion_criteria.py` | Summary / metrics |
| `preferred_metrics.csv` | `preferred_metrics.py` | Summary / metrics |
| `preferred_metrics.csv` | `rebalance_calendar.py` | Summary / metrics |
| `preferred_metrics.csv` | `research_hygiene.py` | Summary / metrics |
| `preferred_metrics.csv` | `risk_metrics_ext.py` | Summary / metrics |
| `preferred_metrics.parquet` | `preferred_metrics.py` | Summary / metrics |
| `preferred_metrics_history.csv` | `fundamentals_history.py` | Summary / metrics |
| `preferred_metrics_history.csv` | `research_hygiene.py` | Summary / metrics |
| `preferred_metrics_history.parquet` | `fundamentals_history.py` | Summary / metrics |
| `price_jump_audit.csv` | `data_integrity.py` | Summary / metrics |
| `rebalance_calendar.csv` | `rebalance_calendar.py` | Summary / metrics |
| `robust_covariance_summary.csv` | `robust_covariance.py` | Summary / metrics |
| `rolling_window_metrics.csv` | `rolling_window_analysis.py` | Summary / metrics |
| `sector_tickers.csv` | `cross_asset_analysis.py` | Summary / metrics |
| `sector_tickers.csv` | `index_registry.py` | Summary / metrics |
| `sharpe_comparison.csv` | `live_index_backtest.py` | Summary / metrics |
| `vix_term_structure_summary.csv` | `vix_term_structure.py` | Summary / metrics |
| `fisher_sector_baskets.csv` | `fisher_sector_baskets.py` | Auxiliary table |
| `fisher_sector_baskets_latest.csv` | `fisher_sector_baskets.py` | Auxiliary table |
| `granger_causality_sectors.csv` | `maintain_analytics.py` | Auxiliary table |
| `growth_tech_membership.csv` | `growth_tech_analytics.py` | Auxiliary table |
| `vix_term_structure.csv` | `vix_term_structure.py` | Auxiliary table |
| `vix_term_structure_live.csv` | `vix_term_structure.py` | Auxiliary table |
| `alerts_log.parquet` | `check_alerts.py` | Other |
| `black_litterman_views.csv` | `black_litterman_views.py` | Other |
| `dupont_analysis.csv` | `dupont_analysis.py` | Other |
| `erc_gmv_strategies.csv` | `portfolio_optimization.py` | Other |
| `fisher_indexes.csv` | `fisher_index.py` | Other |
| `fisher_indexes_duckdb.csv` | `run_fisher_duckdb.py` | Other |
| `fisher_rate_decomposition.csv` | `fisher_index.py` | Other |
| `growth_ai_vol_vs_risk_parity.csv` | `risk_parity_analytics.py` | Other |
| `growth_tech_risk_models.csv` | `growth_tech_analytics.py` | Other |
| `growth_tech_vol_returns.csv` | `growth_tech_analytics.py` | Other |
| `kelly_parameters.parquet` | `kelly.py` | Other |
| `kelly_parameters.parquet` | `preferred_metrics.py` | Other |
| `monte_carlo_terminal_wealth.csv` | `monte_carlo.py` | Other |
| `tail_risk_hedge_crisis.csv` | `tail_risk_hedging.py` | Other |
| `vol_target_vs_risk_parity.csv` | `risk_parity_analytics.py` | Other |
| `vol_targets.csv` | `preferred_metrics.py` | Other |
| `vol_targets.csv` | `vol_target.py` | Other |
| `vol_targets.parquet` | `vol_target.py` | Other |
| `peer_analytics_signals.csv` | `peer_analytics.py` | Screen / decision |
| `peer_group_summary.csv` | `peer_analytics.py` | Summary / metrics |
| `peer_fundamental_trends.csv` | `peer_analytics.py` | Summary / metrics |
| `peer_recovery_signals.csv` | `peer_analytics.py` | Screen / decision |
| `earnings_calendar.parquet` | `update_earnings.py` | Earnings |
| `fundamentals.parquet` | `update_fundamentals.py` | Base parquet table |
| `fundamentals.parquet` | `backfill_edgar.py` | Base parquet table |
| `earnings_catalyst_signals.csv` | `earnings_catalyst.py` | Earnings |
| `earnings_drift_stats.csv` | `earnings_catalyst.py` | Earnings |
| `pair_engine_pairs.csv` | `pair_engine.py` | Pair engine |
| `pair_engine_trades.csv` | `pair_engine.py` | Pair engine |
| `pair_engine_stats.csv` | `pair_engine.py` | Pair engine |
| `cross_section_rankings.csv` | `cross_section.py` | Cross-section |
| `cross_section_returns.csv` | `cross_section.py` | Cross-section |
| `cross_section_stats.csv` | `cross_section.py` | Cross-section |
| `signal_aggregator_scores.csv` | `signal_aggregator.py` | Aggregate |
| `signal_aggregator_scores_history.parquet` | `signal_aggregator.py` | Aggregate (append-only PIT) |
| `signal_aggregator_ic.csv` | `signal_aggregator.py` | Aggregate |
| `regime_forecast_stats.csv` | `regime_forecast.py` | Forecast / anomaly |
| `regime_model_oos.csv` | `pass6.py` | Forecast / anomaly |
| `regime_model_best.csv` | `pass6.py` | Forecast / anomaly |
| `regime_model_matrix.csv` | `pass7.py` | Forecast / anomaly |
| `regime_model_matrix_summary.csv` | `pass7.py` | Forecast / anomaly |
| `regime_model_oos_rpt.csv` | `pass8.py` | Forecast / anomaly |
| `regime_model_best_rpt.csv` | `pass8.py` | Forecast / anomaly |
| `rpt_vs_ibm_compare.csv` | `pass8.py` | Forecast / anomaly |
| `regime_calibration.csv` | `regime_calibrate.py` | Forecast / anomaly |
| `checkpoints/regime/*.pt` | `pass6.py --ckpt-dir` | Forecast / anomaly |
| `technical_signals.csv` | `technical_signals.py` | Technical |
| `economic_calendar.csv` | `economic_calendar.py` | Calendar / events |
|| `estimate_revisions.parquet` | `estimate_revisions.py` | Fundamental |
|| `filings_sentiment.csv` | `filings_sentiment.py` | Sentiment / alternative |
| `ticker_news.parquet` | `ticker_news.py` | Sentiment / alternative |
| `ticker_news_articles.parquet` | `ticker_news.py` | Sentiment / alternative |
| `ticker_news_mentions.parquet` | `ticker_news.py` | Sentiment / alternative |
| `ticker_news_notes.parquet` | `ticker_news.py --notes` | Sentiment / alternative |
|| `options_skew.parquet` | `options_skew.py` | Options |
|| `signal_model_oos.csv` | `signal_model.py` | Aggregate |
|| `signal_model_weights.csv` | `signal_model.py` | Aggregate |
| `shadow_book.csv` | `shadow_book.py` | Portfolio / risk |
| `shadow_lots.csv` | `shadow_book.py` | Portfolio / risk |
| `tail_index.csv` | `tail_index.py` | Taleb / fat tails |
| `portfolio_tail.csv` | `tail_index.py` | Taleb / fat tails |
| `tail_dependence.csv` | `tail_index.py` | Taleb / fat tails |
| `gap_risk.csv` | `gap_risk.py` | Taleb / fat tails |
| `gap_events.csv` | `gap_risk.py` | Taleb / fat tails |
|| `ergodicity_ruin.csv` | `ergodicity_ruin.py` | Taleb / fat tails ||
|| `portfolio_ergodic.csv` | `ergodicity_ruin.py` | Taleb / fat tails ||
|| `fragility_screen.csv` | `fragility_screen.py` | Taleb / fat tails ||
|| `fragility_screen_history.parquet` | `fragility_screen.py` | Taleb / fat tails (append-only PIT) ||
|| `options_skew_history.parquet` | `options_skew.py` | Options (append-only PIT; not back-fillable) ||
|| `macro_fragility.csv` | `macro_fragility.py` | Taleb / fat tails ||
||| `macro_shock.parquet` | `macro_shock.py` | Taleb / fat tails ||
||| `macro_sector_shock.parquet` | `macro_sector_shock.py` | Taleb / fat tails (DYNAMIC baskets) ||
||| `basket_members.parquet` | `macro_sector_shock.py` | Taleb / fat tails ||
|| `shock_ride.parquet` | `shock_ride.py` | Taleb / fat tails ||
|| `shock_ride_tickers.parquet` | `shock_ride.py` | Taleb / fat tails. Per-ticker ride: research momentum (tsmom_*, stmom_1m_ret, gw_high_prox, young_gate_*), fresh breakout (fresh_verdict, fresh_score), fractal (fractal_{90,30}_consensus, fractal_{90,30,15,45}_best_confirmed, fractal_posture, fractal_stack_depth/full/mom), durability (long_ride_score), quality gate (ride_gate_open/horizon/mom), dual exit (ride_exit_flag/kind), structural gate (structural_mode/signal/gate_open/in_market, structural_{turtle,volscale,regime,recouple,consensus}), recommendation/interpretation ||
|| `arista_metrics.parquet` | `arista.py` | Taleb / fat tails. FULL daily ARISTA top-detector metrics time series per ticker (backtesting surface): date,ticker,close,mom3,mom6,decel,downshare,from20,at_year_high,leg_divergence,leg_distribution,leg_rollover,arista_score,arista_signal ||
|| `arista_signals.parquet` | `arista.py` | Taleb / fat tails. LATEST ARISTA snapshot per ticker: date + last-row metrics + arista_signal + interpretation ||
|| `arista_backtest.parquet` | `arista.py` | Taleb / fat tails. Per-ticker signal→forward-120d stats: n_signals, mean_score, avg_fwd_dd, avg_fwd_ret, caught_rate; plus TOTAL row ||
|| `backtest_rides.parquet` | `backtest_rides.py` | Taleb / fat tails. Per-ticker A/B of classic vs quality_gate vs dual_exit vs quality_dual ride strategies ||
|| `backtest_structural.parquet` | `backtest_structural.py` | Taleb / fat tails. Per-ticker daily backtest of structural/risk-scaled gate paradigms: turtle, volscale, regime, recouple, momentum, hybrid, consensus vs buy_hold (ride_return, excess, max_dd_ride, in_market) ||
|| `ride_history.parquet` | `ride_history.py` | Taleb / fat tails. Point-in-time recommended ride trade history per ticker: as_of, mom3/mom12, posture, stack_depth, long_ride_score, ride_gate_open/horizon/mom, ride_exit_flag/kind, recommendation ||
|| `fractal_profiles.parquet` | `statistical_profiler.py` | Taleb / fat tails. True statistical profiler over fractal windows: long-format (ticker, date, span_from, span_to, span_len, close) + price stats (price_mean/median/mode/max/min/range/std/skew/kurtosis/slope/curvature), position stats (close_z/close_pctile/runup/window_drawdown), volume stats (volume_mean/vwap/volume_z), momentum stats (log_ret/momentum/ret_vol) ||
|| `backtest_price_vs_momentum.parquet` | `backtest_price_vs_momentum.py` | Taleb / fat tails. Do price-based fractal signals predict forward returns better than momentum-based? Per feature/horizon: hit_rate_on, mean_on/off, spread, annual_spread, base_mean ||
|| `backtest_long_hold_entry.parquet` | `backtest_long_hold_entry.py` | Taleb / fat tails. Long-hold ignition entry: per ticker excess_t0 vs excess_fair, EW CAGR vs universe, event-study forward excess ||
|| `long_hold_entry_screen.parquet` | `backtest_long_hold_entry.py` | Taleb / fat tails. Recent gap+volume ignitions joined to shock_ride posture/stack/gate/rec ||
|| `rare_ignition_info.parquet` | `rare_ignition_info.py` | Taleb / fat tails. Event-study: raw/fresh/rare/exuberant gap+vol forward excess vs EW ||
|| `rare_ignition_live.parquet` | `rare_ignition_info.py` | Taleb / fat tails. Live rare/exuberant ignitions joined to preferred_metrics + shock_ride ||
||| `fractal_windows_backtest.parquet` | `fractal_windows_backtest.py` | Taleb / fat tails ||
||| `fractal_windows_backtest.parquet` | `fractal_windows_backtest_gpu.py` | Taleb / fat tails ||
||| `subindustry_regime.parquet` | `subindustry_regime.py` | Taleb / fat tails ||
||| `subindustry_regime_lead.parquet` | `subindustry_regime.py` | Taleb / fat tails ||
||| `barbell_check.parquet` | `barbell_check.py` | Taleb / fat tails ||
||| `hidden_optionality.parquet` | `hidden_optionality_audit.py` | Taleb / fat tails ||
|| `corporate_actions.parquet` | `acquisition_backfill.py` | Corporate actions |
|| `fiscal_year_end_map.parquet` | `acquisition_backfill.py` | Fiscal year map |
|| `daily_prices_expired.parquet` | `data_validation.py` | Expired price history |
|| `quarterly_lookthrough_fundamentals.parquet` | `lookthrough_engine.py` | Lookthrough / pro forma |
|| `quarterly_lookthrough_fundamentals_extended.parquet` | `lookthrough_engine.py` | Lookthrough / pro forma |
|| `lookthrough_materialized.parquet` | `lookthrough_materialized.py` | Lookthrough / pro forma |
|| `backfill_checkpoints/*.json` | `resumable_job.py` | Checkpoints (resumable) |
|| `daily_prices_partitioned/` | `partition_daily_prices.py` | Partitioned daily prices |
|| `rolling_window_metrics.parquet` | `rolling_window_analysis.py` | Summary / metrics |
|| `preferred_metrics.parquet` | `preferred_metrics.py` | Summary / metrics |
|| `erp_history.parquet` | `erp_service.py` | Base parquet table |
|| `crp_by_country.parquet` | `damodaran_data.py` | Base parquet table |

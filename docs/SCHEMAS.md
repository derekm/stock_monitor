# Output Schemas (catalog)

This document is the **single source of truth for output file schemas** produced by the stock_monitor programs. Rather than repeat column definitions in every program doc, each output below is assigned a **schema family**; program docs link here. Column lists under each family are *representative of that family's shape* — open the producing script for the exact DataFrame built.


## Schema families

### Correlation matrix  (`correlation_matrix`)

- Shape: `index` (row label, often `ticker` or `sector`), then one column per entity (ticker/sector) holding the pairwise correlation coefficient in [-1, 1]. Square (N×N) or long-form `a,b,corr`.

### Regime / state table  (`regime_state`)

- Shape: `date` (or `as_of`), `regime`/`state` (label e.g. Calm/Stress), and state probabilities or latent estimates (`p_calm`,`p_stress`, `mkt`, `log_vol`, `kalman_state`, `entropy`). One row per trading day.

### Weights / performance  (`weights_performance`)

- Shape: Strategy/name-level tables: `ticker` (or `strategy`), `weight` (fraction), plus risk/return stats (`ret`, `vol`, `sharpe`, `max_dd`) and per-name `rc` (risk contribution) where relevant. One row per name or per (date, name).

### Forecast / anomaly  (`forecast_anomaly`)

- Shape: Forecast rows: `ticker`, `as_of`/`forecast_date`, `horizon`, `pct_change` (and `close`/`history` for charts); anomaly rows: `ticker`, `date`, `z_*` scores, `flag`.

### Screen / decision  (`screen_decision`)

- Shape: Decision tables keyed by `ticker`: boolean/label columns for each gate leg (`roe`,`roic`,`debt_to_equity`,`ev_ebitda`,`pb_ratio`,`mktcap_to_assets`), a `decision`/`action` label, and `fail_legs`. May include `earnings_stability`, `composite_score`, `w_max`.

### Index level series  (`index_levels`)

- Shape: Time series: `date`, `level` (index value, base 100), and component `return` columns. Stored as Parquet (and sometimes CSV). One row per trading day.

### Base parquet table  (`base_table`)

- Shape: Canonical parquet inputs: `daily_prices` (date,ticker,open,high,low,close,volume,adj_close), `fundamentals` (ticker,as_of_date,market_cap_b,...,pb_ratio,ev_ebitda,mktcap_to_assets,...), `monitored_stocks` (ticker,sector,index_member,...), `portfolio_holdings`, `trades`.

### Summary / metrics  (`summary_metrics`)

- Shape: One-row or few-row aggregates: `_summary`/`_stats`/`_metrics` with scalars (counts, vol, sharpe, avg corr, pass counts, reliability ranks). Long-form `name,value` also common.

### Auxiliary table  (`aux_table`)

- Shape: Supporting tables: `sector_tickers` (ticker,sector,SECT_* slug), `vix_term_structure` (tenor, iv), membership/catalog listings.

### Other  (`other`)

- Shape: See producing script.


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
| `hmm_regime_states.csv` | `hmm_posterior_analysis.py` | Regime / state table |
| `hmm_regime_states.csv` | `hmm_regime_detection.py` | Regime / state table |
| `hmm_regime_states.csv` | `kalman_gain_analysis.py` | Regime / state table |
| `hmm_regime_states.csv` | `kalman_state_estimates.py` | Regime / state table |
| `hmm_regime_states.csv` | `monte_carlo.py` | Regime / state table |
| `hmm_regime_states.csv` | `posterior_entropy_dynamics.py` | Regime / state table |
| `hmm_regime_states.csv` | `regime_aware_constraints.py` | Regime / state table |
| `hmm_regime_states.csv` | `regime_correlation_breakdown.py` | Regime / state table |
| `hmm_regime_states.csv` | `threshold_logic.py` | Regime / state table |
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
| `momentum_ic.csv` | `momentum_analytics.py` | Screen / decision |
| `momentum_metrics.csv` | `buy_candidates.py` | Screen / decision |
| `momentum_metrics.csv` | `factor_panel.py` | Screen / decision |
| `momentum_metrics.csv` | `momentum_analytics.py` | Screen / decision |
| `momentum_quintiles.csv` | `momentum_analytics.py` | Screen / decision |
| `near_dual_candidates.csv` | `inclusion_criteria.py` | Screen / decision |
| `portfolio_risk_summary.csv` | `risk_metrics_ext.py` | Screen / decision |
| `preferred_screen_hits.csv` | `preferred_metrics.py` | Screen / decision |
| `risk_metrics_ext.csv` | `buy_candidates.py` | Screen / decision |
| `risk_metrics_ext.csv` | `risk_metrics_ext.py` | Screen / decision |
| `rolling_screen_stability.csv` | `rolling_window_analysis.py` | Screen / decision |
| `screen_backtest.csv` | `fundamentals_history.py` | Screen / decision |
| `threshold_logic_screen.csv` | `threshold_logic.py` | Screen / decision |
| `daily_prices.parquet` | `allpairs_correlations.py` | Index level series |
| `daily_prices.parquet` | `analyze_granite_forecasts.py` | Index level series |
| `daily_prices.parquet` | `backfill_constituents.py` | Index level series |
| `daily_prices.parquet` | `backfill_historical.py` | Index level series |
| `daily_prices.parquet` | `binding_constraints_analysis.py` | Index level series |
| `daily_prices.parquet` | `build_defensive_index.py` | Index level series |
| `daily_prices.parquet` | `build_growth_tech_index.py` | Index level series |
| `daily_prices.parquet` | `build_index.py` | Index level series |
| `daily_prices.parquet` | `check_alerts.py` | Index level series |
| `daily_prices.parquet` | `crisis_correlation.py` | Index level series |
| `daily_prices.parquet` | `cross_asset_analysis.py` | Index level series |
| `daily_prices.parquet` | `data_access.py` | Index level series |
| `daily_prices.parquet` | `data_integrity.py` | Index level series |
| `daily_prices.parquet` | `data_integrity_deep.py` | Index level series |
| `daily_prices.parquet` | `factor_rotation_defense.py` | Index level series |
| `daily_prices.parquet` | `fisher_index.py` | Index level series |
| `daily_prices.parquet` | `forecast_granite.py` | Index level series |
| `daily_prices.parquet` | `growth_tech_analytics.py` | Index level series |
| `daily_prices.parquet` | `hmm_regime_detection.py` | Index level series |
| `daily_prices.parquet` | `inclusion_criteria.py` | Index level series |
| `daily_prices.parquet` | `kalman_gain_analysis.py` | Index level series |
| `daily_prices.parquet` | `kalman_state_estimates.py` | Index level series |
| `daily_prices.parquet` | `maintain_analytics.py` | Index level series |
| `daily_prices.parquet` | `monte_carlo.py` | Index level series |
| `daily_prices.parquet` | `portfolio_optimization.py` | Index level series |
| `daily_prices.parquet` | `portfolio_report.py` | Index level series |
| `daily_prices.parquet` | `rebalance_calendar.py` | Index level series |
| `daily_prices.parquet` | `regime_aware_constraints.py` | Index level series |
| `daily_prices.parquet` | `regime_correlation_breakdown.py` | Index level series |
| `daily_prices.parquet` | `risk_metrics_ext.py` | Index level series |
| `daily_prices.parquet` | `risk_parity_analytics.py` | Index level series |
| `daily_prices.parquet` | `robust_covariance.py` | Index level series |
| `daily_prices.parquet` | `rolling_correlation_windows.py` | Index level series |
| `daily_prices.parquet` | `rolling_window_analysis.py` | Index level series |
| `daily_prices.parquet` | `run_fisher_duckdb.py` | Index level series |
| `daily_prices.parquet` | `tail_risk_hedging.py` | Index level series |
| `daily_prices.parquet` | `tspulse_anomaly.py` | Index level series |
| `daily_prices.parquet` | `ttm_exogenous.py` | Index level series |
| `daily_prices.parquet` | `ttm_features.py` | Index level series |
| `daily_prices.parquet` | `update_prices.py` | Index level series |
| `daily_prices.parquet` | `vix_term_structure.py` | Index level series |
| `daily_prices.parquet` | `vol_target.py` | Index level series |
| `daily_prices_clean.parquet` | `data_integrity.py` | Index level series |
| `daily_prices_clean.parquet` | `data_integrity_deep.py` | Index level series |
| `daily_prices_yfinance.parquet` | `backfill_constituents.py` | Index level series |
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
| `sector_prices.parquet` | `cross_asset_analysis.py` | Index level series |
| `sector_prices.parquet` | `data_access.py` | Index level series |
| `sector_prices.parquet` | `forecast_granite.py` | Index level series |
| `sector_prices.parquet` | `index_registry.py` | Index level series |
| `alerts_config.parquet` | `alerts_config.parquet.py` | Base parquet table |
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
| `exogenous_panel.parquet` | `tspulse_anomaly.py` | Other |
| `exogenous_panel.parquet` | `ttm_exogenous.py` | Other |
| `fisher_indexes.csv` | `fisher_index.py` | Other |
| `fisher_indexes_duckdb.csv` | `run_fisher_duckdb.py` | Other |
| `fisher_rate_decomposition.csv` | `fisher_index.py` | Other |
| `growth_ai_vol_vs_risk_parity.csv` | `risk_parity_analytics.py` | Other |
| `growth_tech_risk_models.csv` | `growth_tech_analytics.py` | Other |
| `growth_tech_vol_returns.csv` | `growth_tech_analytics.py` | Other |
| `kelly_parameters.parquet` | `kelly.py` | Other |
| `kelly_parameters.parquet` | `preferred_metrics.py` | Other |
| `monte_carlo_terminal_wealth.csv` | `monte_carlo.py` | Other |
| `sp500_universe_tracking.parquet` | `sp_universe_tracking.py` | Other |
| `tail_risk_hedge_crisis.csv` | `tail_risk_hedging.py` | Other |
| `vol_target_vs_risk_parity.csv` | `risk_parity_analytics.py` | Other |
| `vol_targets.csv` | `preferred_metrics.py` | Other |
| `vol_targets.csv` | `vol_target.py` | Other |
| `vol_targets.parquet` | `vol_target.py` | Other |

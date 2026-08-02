# CSV → generating program map

| CSV | Program |
|-----|---------|
| preferred_metrics.csv / preferred_screen_hits.csv | preferred_metrics.py |
| preferred_metrics_history.csv / screen_backtest.csv | fundamentals_history.py |
| dual_pass_stress.csv / dual_pass_sensitivity.csv | stress_dual_pass.py |
| inclusion_candidates / exclusion / near_dual / defensive_value_exploration | inclusion_criteria.py |
| binding_*.csv | binding_constraints_analysis.py |
| threshold_logic_screen.csv / threshold_logic_rules.json | threshold_logic.py |
| regime_aware_*.csv / regime_constraint_binding.csv | regime_aware_constraints.py |
| hmm_regime_*.csv / hmm_transition_matrix.csv | hmm_regime_detection.py |
| hmm_posterior_*.csv / hmm_uncertain_days.csv | hmm_posterior_analysis.py |
| posterior_entropy_*.csv | posterior_entropy_dynamics.py |
| hmm_transition_triggers.csv | regime_aware_constraints.py |
| regime_corr_*.csv / regime_sector_corr.csv | regime_correlation_breakdown.py |
| crisis_correlation_*.csv | crisis_correlation.py |
| rolling_corr_*.csv / rolling_sector_corr_windows.csv | rolling_correlation_windows.py |
| rolling_window_metrics.csv / rolling_screen_stability.csv | rolling_window_analysis.py |
| allpairs_*.csv | allpairs_correlations.py |
| asset_correlation_matrix.csv / sector_correlation_matrix_latest.csv | inclusion_criteria.py |
| factor_rotation_*.csv / factor_sleeve_returns.csv | factor_rotation_defense.py |
| tail_risk_hedge_*.csv | tail_risk_hedging.py |
| kalman_state_*.csv | kalman_state_estimates.py |
| kalman_gain_*.csv | kalman_gain_analysis.py |
| vix_term_structure*.csv | vix_term_structure.py |
| risk_metrics.csv | risk_enrich.py |
| dupont_analysis.csv | dupont_analysis.py |
| vol_target_vs_risk_parity.csv / vol_targets.csv | vol_target.py / risk_parity_analytics.py |
| erc_gmv_*.csv | portfolio_optimization.py |
| black_litterman_weights.csv | black_litterman.py |
| robust_covariance_summary.csv | robust_covariance.py |
| growth_tech_*.csv | growth_tech_analytics.py / build_growth_tech_index.py |
| fisher_indexes*.csv | fisher_index.py / run_fisher_duckdb.py |
| forecasts_granite.csv | forecast_granite.py |
| anomalies_tspulse.csv | tspulse_anomaly.py |
| defensive_value_etfs.csv | monitor maintenance seeds |

Master: `run_daily_automation.py`

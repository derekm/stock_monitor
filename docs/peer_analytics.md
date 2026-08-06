# peer_analytics.py

Cross-stock peer comparison analytics — automates the RF-style deep-dive across all stocks by comparing each name to its industry sector and analytics-group peers, revealing fundamental trends, risk-adjusted rankings, and recovery patterns.

## Why it exists (rationale)

The RF analysis showed that a single stock's "underperformance" is only meaningful relative to peers and its own history. This module scales that logic to the full universe:

- **Peer-relative scoring** — every stock ranked within its sector, sleeve, and index group
- **Fundamental trend detection** — linear slopes for ROE, ROIC, earnings stability, P/B, leverage, valuation
- **Recovery / deterioration signals** — metrics that declined then improved (or vice versa)
- **Risk-adjusted peer ranks** — return, vol, and Sharpe percentiles within each group
- **Beta to peer group** — identifies high/low sector beta names

Outputs feed directly into the daily pipeline, dashboard, and regime-aware rebalancing.

## Usage

```bash
python peer_analytics.py --save
```

Flags: `--save` (default true). Reads `daily_prices.parquet`, `fundamentals.parquet`, `monitored_stocks.parquet`.

## Peer group construction

Groups are built from `monitored_stocks.parquet` columns:

| Source column | Group prefix | Min size | Example groups |
|---|---|---|---|
| `sector` | `sector_` | 3 | `sector_Financials` (19), `sector_Information Technology` (20) |
| `growth_sleeve` | `sleeve_` | 3 | `sleeve_growth_ai` (5), `sleeve_starlink_supply` (11) |
| `value_sleeve` | `value_` | 3 | `value_defensive_etf` (12) |
| `defensive_value_index` | — | 3 | `defensive_index` (82) |
| `growth_tech_index` | — | 3 | `growth_tech_index` (43) |

Total: **25 peer groups** covering all 142 tracked names (sector + sleeve + index membership).

## Fundamental trend metrics tracked

For each ticker with ≥4 fundamental observations, computes OLS slope per period:

| Metric | What it reveals |
|---|---|
| `roe` | Profitability trend (RF: 44% → 15% = deterioration) |
| `roic` | Capital efficiency trend |
| `earnings_stability` | Earnings quality trend (RF: 16.1 → 0.72 = collapse) |
| `pb_ratio` | Valuation expansion/contraction (RF: 0.85 → 1.10 while ROE fell) |
| `debt_to_equity` | Leverage trend |
| `ev_ebitda` | Valuation vs earnings trend |
| `mktcap_to_assets` | Market vs book trend |
| `interest_coverage` | Debt service capacity trend |

Outputs per metric: `slope_per_period`, `total_change`, `pct_change`, `recent_vs_early_pct`, `latest_value`, `n_obs`.

## Recovery / deterioration detection

A metric is flagged as **recovering** if:
- Overall slope is negative (deteriorating long-term)
- But recent (last 2 periods) vs early (first 2) shows >5% improvement

A metric is flagged as **deteriorating** if:
- Overall slope is positive (improving long-term)
- But recent vs early shows >5% decline

**RF example**: ROE overall slope negative (44%→15%), but recent vs early depends on latest quarters.

## Peer-relative rankings

For each stock in each peer group, computes percentiles at 63d/126d/252d windows:

| Rank | Meaning |
|---|---|
| `ret_rank` | Percentile of recent return vs peers (higher = better) |
| `vol_rank` | Percentile of recent vol vs peers (lower = better) |
| `sharpe_rank` | Percentile of risk-adjusted return (higher = better) |

**Best peer group** per stock = group where it has highest `sharpe_rank` (126d window primary).

## Beta to peer group

Rolling 126d beta of each stock to its group's equal-weight return:

| Signal | Threshold |
|---|---|
| `high_beta_flag` | Beta > group 75th percentile |
| `low_beta_flag` | Beta < group 25th percentile |

**RF example**: Beta to XLF (Financials) = 1.19 (high beta, amplifies sector stress).

## Composite signals generated

| Signal column | Logic |
|---|---|
| `fundamental_signal` | `RECOVERING` (≥2 recovering metrics) / `DETERIORATING` (≥2 deteriorating) / `STRONG_TREND` (≥3 strong consistent trends) / `NEUTRAL` |
| `peer_signal` | `PEER_LEADER` (sharpe_rank ≥ 0.75) / `PEER_LAGGARD` (≤ 0.25) / `PEER_AVERAGE` |
| `beta_signal` | `HIGH_BETA` / `LOW_BETA` / `NEUTRAL_BETA` |

Plus counts: `recovery_count`, `deterioration_count`, `strong_trend_count`.

## Outputs

| File | Schema family | Description |
|---|---|---|
| `peer_analytics_signals.csv` | `screen_decision` | Per-ticker composite signals + latest fundamentals + best peer group ranks + beta |
| `peer_group_summary.csv` | `summary_metrics` | Group-level stats (mean, median, std, p25, p75) for each fundamental metric |
| `peer_fundamental_trends.csv` | `summary_metrics` | Per-ticker, per-metric trend slopes and changes |
| `peer_recovery_signals.csv` | `screen_decision` | Per-ticker, per-metric recovery/deterioration/strong-trend flags |

## Pipeline integration

1. **Daily automation**: Add `peer` to `run_daily_automation.py` JOBS list
2. **Dashboard**: Tables auto-loaded by `export_dashboard_data.py` (added to TABLES list)
3. **SCHEMAS.md**: Registered in output catalog
4. **Regime-aware consumers**: Signals available for `regime_aware_constraints`, `portfolio_optimization`, `rebalance_calendar`

## Example: RF's signals (as computed)

| Signal | Value | Interpretation |
|---|---|---|
| `fundamental_signal` | `DETERIORATING` | ROE, ROIC, earnings_stability all declining |
| `peer_signal` | `PEER_LAGGARD` | Sharpe rank low vs Financials peers |
| `beta_signal` | `HIGH_BETA` | Beta 1.19 to XLF (75th pctl) |
| `recovery_count` | 0 | No metrics recovering |
| `deterioration_count` | 3+ | ROE, ROIC, earnings_stability, pb_ratio |
| `best_peer_group` | `sector_Financials` | Where it ranks (poorly) |
| `best_sharpe_rank` | ~0.1-0.2 | Bottom quartile vs peers |

## Related programs

- **Inputs**: `daily_prices.parquet`, `fundamentals.parquet`, `monitored_stocks.parquet`
- **Consumers**: `export_dashboard_data.py`, `run_daily_automation.py` (add to JOBS)
- **Complements**: `preferred_metrics.py` (dual-pass screen), `threshold_logic.py` (regime thresholds), `rolling_window_analysis.py` (portfolio-level metrics)
- **Documentation**: [SCHEMAS.md](SCHEMAS.md), [SYSTEM_ORCHESTRATION.md](SYSTEM_ORCHESTRATION.md)
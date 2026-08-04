# kalman_state_estimates.py

kalman_state_estimates.py — Kalman filter latent state for market risk.

## Why it exists (rationale)

Kalman-filter latent market-risk state used by `regime_correlation_breakdown`, `maintain_analytics` (kalman_correlations), and the dashboard.

## Usage

```bash
python kalman_state_estimates.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
  - `kalman_state_estimates.csv`
  - `kalman_state_summary.csv`


## Related programs

- [docs/kalman_gain_analysis.md](kalman_gain_analysis.md)
- [docs/regime_correlation_breakdown.md](regime_correlation_breakdown.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

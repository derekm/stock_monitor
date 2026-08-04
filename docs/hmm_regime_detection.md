# hmm_regime_detection.py

hmm_regime_detection.py — Gaussian HMM regimes on market returns + vol.

## Why it exists (rationale)

Fits a Gaussian HMM to market returns + vol, emitting regime states/transitions that `regime_aware_constraints`, `regime_correlation_breakdown`, and `monte_carlo` consume.

## Usage

```bash
python hmm_regime_detection.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
  - `hmm_regime_summary.csv`
- **Correlation matrix** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_transition_matrix.csv`


## Related programs

- [docs/regime_correlation_breakdown.md](regime_correlation_breakdown.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/monte_carlo.md](monte_carlo.md)
- [docs/kalman_state_estimates.md](kalman_state_estimates.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

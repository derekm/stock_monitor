# monte_carlo.py

monte_carlo.py — Regime-switching Monte Carlo with variance-reduction options.

## Why it exists (rationale)

Regime-switching Monte Carlo with variance reduction (Sobol) for tail/wealth simulation; uses HMM regimes from `hmm_regime_detection`.

## Usage

```bash
python monte_carlo.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
- **Correlation matrix** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_transition_matrix.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monte_carlo_path_stats.csv`
  - `monte_carlo_summary.csv`
- **Other** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monte_carlo_terminal_wealth.csv`


## Related programs

- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/mcmc_regimes.md](mcmc_regimes.md)
- [docs/sobol_qmc.md](sobol_qmc.md)
- [docs/tail_risk_hedging.md](tail_risk_hedging.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

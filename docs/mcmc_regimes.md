# mcmc_regimes.py

mcmc_regimes.py — Lightweight MCMC for regime-conditional return means.

## Why it exists (rationale)

Lightweight MCMC over regime-conditional return means — a Bayesian cross-check on `hmm_regime_detection` / `monte_carlo` regime assumptions.

## Usage

```bash
python mcmc_regimes.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `mcmc_regime_means.csv`
  - `mcmc_regime_summary.csv`
  - `mcmc_transition_draws.csv`


## Related programs

- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/monte_carlo.md](monte_carlo.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

# posterior_entropy_dynamics.py

posterior_entropy_dynamics.py — Dynamics of HMM posterior entropy.

## Why it exists (rationale)

Tracks HMM posterior entropy dynamics as a regime-uncertainty signal for `regime_aware_constraints` and the dashboard.

## Usage

```bash
python posterior_entropy_dynamics.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_posterior_analysis.csv`
  - `hmm_regime_states.csv`
  - `posterior_entropy_dynamics.csv`
  - `posterior_entropy_summary.csv`


## Related programs

- [docs/hmm_posterior_analysis.md](hmm_posterior_analysis.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

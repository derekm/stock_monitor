# hmm_posterior_analysis.py

hmm_posterior_analysis.py — Explore HMM hidden-state posterior probabilities.

## Why it exists (rationale)

Explores HMM hidden-state posterior probabilities (uncertainty, entropy) that `regime_aware_constraints` and `posterior_entropy_dynamics` build on.

## Usage

```bash
python hmm_posterior_analysis.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_posterior_analysis.csv`
  - `hmm_posterior_summary.csv`
  - `hmm_regime_states.csv`
  - `hmm_uncertain_days.csv`


## Related programs

- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/posterior_entropy_dynamics.md](posterior_entropy_dynamics.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

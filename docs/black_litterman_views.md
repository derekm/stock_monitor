# black_litterman_views.py

black_litterman_views.py — Build BL views from dual-pass / regime posture.

## Why it exists (rationale)

Turns dual-pass posture + regime state into Black-Litterman view vectors consumed by `black_litterman.py`, closing the loop from screens to posterior weights.

## Usage

```bash
python black_litterman_views.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Other** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `black_litterman_views.csv`
- **Weights / performance** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `black_litterman_weights_from_views.csv`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regimes.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`


## Related programs

- [docs/black_litterman.md](black_litterman.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

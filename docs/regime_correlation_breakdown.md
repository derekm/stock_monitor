# regime_correlation_breakdown.py

regime_correlation_breakdown.py — Correlation structure inside each HMM regime.

## Why it exists (rationale)

Correlation structure inside each HMM regime (pair deltas, sector corr) — the regime-conditioned view used by `regime_aware_constraints`.

## Usage

```bash
python regime_correlation_breakdown.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monitored_stocks.parquet`
- **Correlation matrix** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `regime_corr_breakdown.csv`
  - `regime_corr_pair_delta.csv`
  - `regime_sector_corr.csv`


## Related programs

- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/kalman_state_estimates.md](kalman_state_estimates.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

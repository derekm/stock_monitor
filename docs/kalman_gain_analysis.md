# kalman_gain_analysis.py

Kalman gain path analysis for (mkt_ret, log_vol) filter.

## Why it exists (rationale)

Diagnoses the Kalman filter gain path for the (market-return, log-vol) latent state — supports `kalman_state_estimates`.

## Usage

```bash
python kalman_gain_analysis.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
  - `kalman_gain_path.csv`
  - `kalman_gain_summary.csv`


## Related programs

- [docs/kalman_state_estimates.md](kalman_state_estimates.md)
- [docs/regime_correlation_breakdown.md](regime_correlation_breakdown.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

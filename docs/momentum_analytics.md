# momentum_analytics.py

momentum_analytics.py — Cross-sectional and time-series momentum.

## Why it exists (rationale)

Cross-sectional + time-series momentum (IC, quintiles, metrics) feeding `factor_panel` and `buy_candidates`.

## Usage

```bash
python momentum_analytics.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `momentum_ic.csv`
  - `momentum_metrics.csv`
  - `momentum_quintiles.csv`


## Related programs

- [docs/factor_panel.md](factor_panel.md)
- [docs/buy_candidates.md](buy_candidates.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

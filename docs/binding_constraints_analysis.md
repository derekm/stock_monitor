# binding_constraints_analysis.py

binding_constraints_analysis.py — Impact of dual-pass binding constraints.

## Why it exists (rationale)

Quantifies how the dual-pass binding constraints actually bind — which baskets carry the most risk and which near-miss names are closest to passing — feeding `inclusion_criteria` tuning.

## Usage

```bash
python binding_constraints_analysis.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `binding_basket_risk.csv`
  - `binding_constraints_impact.csv`
  - `binding_near_miss_detail.csv`
- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`


## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

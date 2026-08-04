# vix_term_structure.py

VIX / vol term-structure exploration (offline realized-vol proxy).

## Why it exists (rationale)

VIX / vol term-structure exploration using an offline realized-vol proxy — a regime/vol signal for `regime_correlation_breakdown` and the dashboard.

## Usage

```bash
python vix_term_structure.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Auxiliary table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `vix_term_structure.csv`
  - `vix_term_structure_live.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `vix_term_structure_summary.csv`


## Related programs

- [docs/regime_correlation_breakdown.md](regime_correlation_breakdown.md)
- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

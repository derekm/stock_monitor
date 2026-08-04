# factor_panel.py

factor_panel.py — multi-factor panel: value, quality, momentum, low-vol, leverage flag.

## Why it exists (rationale)

Builds the multi-factor panel (value, quality, momentum, low-vol, leverage flag) that `buy_candidates`, `momentum_analytics`, and `preferred_metrics` consume.

## Usage

```bash
python factor_panel.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `factor_panel.csv`
  - `factor_panel_top.csv`
  - `momentum_metrics.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`


## Related programs

- [docs/buy_candidates.md](buy_candidates.md)
- [docs/momentum_analytics.md](momentum_analytics.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

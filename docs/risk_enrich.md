# risk_enrich.py

Add realized vol, beta, max DD to preferred_metrics and fundamentals analytics.

## Why it exists (rationale)

Adds realized vol, beta, max-DD to preferred_metrics / fundamentals analytics — risk features for `risk_metrics_ext` and sizing.

## Usage

```bash
python risk_enrich.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/risk_metrics_ext.md](risk_metrics_ext.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

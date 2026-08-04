# quality_gate_bridge.py

quality_gate_bridge.py — canonical dual-screen quality/value gate for stock_monitor.

## Why it exists (rationale)

Canonical dual-screen quality/value gate for stock_monitor.

## Usage

```bash
python quality_gate_bridge.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/threshold_logic.md](threshold_logic.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

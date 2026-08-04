# manage_alerts.py

manage_alerts.py - Add, enable/disable, or list alert rules.

## Why it exists (rationale)

manage_alerts.py - Add, enable/disable, or list alert rules.

## Usage

```bash
python manage_alerts.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `alerts_config.parquet`


## Related programs

- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

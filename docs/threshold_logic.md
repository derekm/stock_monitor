# threshold_logic.py

threshold_logic.py — Reusable dual-pass / regime-aware threshold logic.

## Why it exists (rationale)

Reusable dual-pass / regime-aware threshold logic (the rule engine) — single source for the gates `inclusion_criteria` applies.

## Usage

```bash
python threshold_logic.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regime_states.csv`
- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `threshold_logic_screen.csv`


## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/quality_gate_bridge.md](quality_gate_bridge.md)
- [docs/stress_dual_pass.md](stress_dual_pass.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

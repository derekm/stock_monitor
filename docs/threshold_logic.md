# threshold_logic.py

Reusable dual-pass / regime-aware threshold logic — the single source of truth
for screen thresholds.

## Why it exists (rationale)

Multiple scripts (inclusion, preferred, stress, regime-constrained) need the same
dual-pass legs and regime-aware relaxations. Centralizing them here means one
authoritative threshold set (BASE legs + regime overrides) instead of drifted
copies. `threshold_logic_screen.csv` is the evaluated output; the module is also
imported by the screen layer.

## Usage

```bash
python threshold_logic.py --save
python threshold_logic.py --regime high_vol_stress --from-hmm
```

Flags: `--regime` (override regime), `--from-hmm` (read regime from
`hmm_regime_states.csv`), `--save`. Reads `fundamentals.parquet`.

## Outputs

- `threshold_logic_screen.csv` — per-ticker leg evaluation under the chosen regime

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [inclusion_criteria.md](inclusion_criteria.md) / [preferred_metrics.md](preferred_metrics.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [stress_dual_pass.md](stress_dual_pass.md)

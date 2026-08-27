# stress_dual_pass.py

Stress-test the dual-pass inclusion criteria.

## Why it exists (rationale)

The dual-pass thresholds are judgment calls. This varies the ROE / ROIC / D/E /
EV/EBITDA / P/B / MktCap/Assets thresholds and reports how many names pass, plus
one-leg relaxation sensitivity — so you can see how fragile the INCLUDE_CORE set
is to each leg and set thresholds with eyes open (pairs with
`binding_constraints_analysis`).

## Usage

```bash
python stress_dual_pass.py --save
```

Flags: `--save`. Reads `fundamentals.parquet`, `daily_prices/`,
`monitored_stocks.parquet`, `preferred_metrics.csv`.

## Outputs

- `dual_pass_stress.csv` — pass counts across threshold grid + one-leg relaxations

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [inclusion_criteria.md](inclusion_criteria.md) — the gate it stresses
- [binding_constraints_analysis.md](binding_constraints_analysis.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [preferred_metrics.md](preferred_metrics.md)

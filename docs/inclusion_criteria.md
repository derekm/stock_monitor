# inclusion_criteria.py

Documented, automated inclusion/exclusion rules — the gate that turns
`preferred_metrics` scores into INCLUDE_CORE / VALUE / QUALITY / SATELLITE /
WATCH / AVOID bands.

## Why it exists (rationale)

Screens need a single authoritative rule set, not ad-hoc thresholds scattered
across scripts. This encodes the dual-pass (Buffett quality AND value trifecta)
plus the softer bands, emits the candidate lists, and writes the rule set to
JSON so the rest of the stack (and the dashboard) reference one source of truth.

## Dual-pass (INCLUDE_CORE) — must satisfy ALL

- Quality: ROE ≥ 0.15, ROIC ≥ 0.15, Debt/Equity ≤ 1.0
- Value: EV/EBITDA ≤ 9.0, P/B ≤ 1.5, MktCap/Assets ≤ 0.5

Bands (from `preferred_metrics` composite): INCLUDE_VALUE (trifecta),
INCLUDE_QUALITY (Buffett), SATELLITE (≥0.50), WATCH (≥0.35), AVOID (else).
Per-name hard cap (default 5%) applies regardless of score.

## Usage

```bash
python inclusion_criteria.py --save
```

Flags: `--save`. Reads `fundamentals.parquet`, `monitored_stocks.parquet`,
`daily_prices/`, `portfolio_holdings.parquet`, `preferred_metrics.csv`.

## Outputs

- `inclusion_candidates.csv` — names passing INCLUDE_CORE
- `exclusion_candidates.csv` — names failing
- `near_dual_candidates.csv` — near-misses (one leg short)
- `defensive_value_exploration.csv` — defensive-value exploration
- `inclusion_rules.json` — the rule set (source of truth)

(Schema families: screen_decision / aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — scores it bands
- [binding_constraints_analysis.md](binding_constraints_analysis.md) — leg impact
- [stress_dual_pass.md](stress_dual_pass.md) — scenario stress
- [dual_screen_analysis.md](dual_screen_analysis.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)

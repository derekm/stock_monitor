# data_integrity.py

Price / fundamental integrity utilities (Polars-first). Audits and repairs the
base tables before they feed analytics.

## Why it exists (rationale)

Bad ticks, unadjusted splits, and schema drift quietly corrupt every downstream
metric. This script is the first-line data-quality gate: detect/clip bad price
jumps, optionally split-adjust via jump detection, build point-in-time
fundamental joins, enforce volume hygiene for Fisher weights, and schema-check
critical artifacts.

## Usage

```bash
python data_integrity.py audit
python data_integrity.py clean-prices --save --clip 0.35
python data_integrity.py pit-fundamentals --save
python data_integrity.py schema-check
```

Subcommands: `audit`, `clean-prices`, `pit-fundamentals`, `schema-check`.
Flags: `--clip` (jump threshold, default 0.35), `--save`.

## Outputs

- `daily_prices_clean.parquet` — clipped/split-adjusted prices
- `fundamentals_pit.parquet` — point-in-time fundamental joins
- `price_jump_audit.csv` — detected jumps
- `schema_check_report.json` — schema conformance report

(Schema families: base_table / aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [data_integrity_deep.md](data_integrity_deep.md) — deeper scan
- [backfill_historical.md](backfill_historical.md) / [update_prices.md](update_prices.md) — sources
- [fundamentals_history.md](fundamentals_history.md)

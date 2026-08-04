# build_defensive_index.py

Equal-weight Defensive / Value index snapshot.

## Why it exists (rationale)

A fourth benchmark sleeve: a low-vol, defensively-positioned book (Staples,
Healthcare/Pharma, Telecom/Utilities, select Industrials) used to compare the
personal portfolio against a conservative alternative and to anchor tail-risk
thinking.

## Usage

```bash
python build_defensive_index.py
```

Flags: none (reads `monitored_stocks.parquet` where `defensive_value_index=True`).
Writes the index parquet and prints P/B, EV/EBITDA, market-cap context.

## Outputs

- `defensive_value_index.parquet` — daily equal-weight index level + component
  returns

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [build_index.md](build_index.md) — fertilizer index
- [build_growth_tech_index.md](build_growth_tech_index.md) — higher-vol sleeve
- [inclusion_criteria.md](inclusion_criteria.md) — `defensive_value_index` flag source
- [manage_stocks.md](manage_stocks.md) — sets the flag

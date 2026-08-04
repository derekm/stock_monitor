# build_growth_tech_index.py

Equal-weight higher-risk Growth / Tech index, built from the `growth_sleeve`
members in `monitored_stocks.parquet`.

## Why it exists (rationale)

A capped, higher-volatility satellite sleeve (alongside fertilizer,
defensive-value, and the personal portfolio) to track a basket of growth/tech
names separately. It is explicitly a **capped satellite**, not the core: sizing
and per-name caps still bind.

## Sleeves (growth_sleeve on monitored_stocks)

- `growth_ai` — NVDA, AMD, PLTR, CRWD (and related high-beta AI names)
- `quality_growth` — MSFT, GOOGL
- `emerging_growth` — TSLA, ENPH, SEDG, REGN, XBI
- `cyclical` — BA, CAT, SCHW
- `thematic` — ARKK, QQQ, VUG (small satellite slice)

Members = `growth_tech_index=True`.

## Usage

```bash
python build_growth_tech_index.py
```

Flags: none (uses `cli_common` resolver only if invoked with `--index`/`--ticker`).
Writes the index + levels parquet.

## Outputs

- `growth_tech_index.parquet` — daily equal-weight index level
- `growth_tech_index_levels.parquet` — component level series

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [growth_tech_analytics.md](growth_tech_analytics.md) — full analysis suite
- [build_index.md](build_index.md) / [build_defensive_index.md](build_defensive_index.md)
- [inclusion_criteria.md](inclusion_criteria.md) / [manage_stocks.md](manage_stocks.md)

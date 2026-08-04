# manage_stocks.py

Maintain the `monitored_stocks.parquet` master table — the source of ticker
metadata and index-membership flags.

## Why it exists (rationale)

Almost every other script resolves tickers/indices from `monitored_stocks`
(via `index_registry`). This is the editor for that master table: add tickers,
set status (active/monitored/inactive), toggle index-membership flags
(`index_member`, `defensive_value_index`, `growth_tech_index`, `dual_pass_member`,
`growth_sleeve`), and apply staged JSON from the dashboard's Manage tab.

## Usage

```bash
python manage_stocks.py list [--status active] [--sector ...]
python manage_stocks.py add --ticker TICK --name "Name" --sector Materials \
       --industry "..." --status active --index_member
python manage_stocks.py set_status TICK active|monitored|inactive
python manage_stocks.py set_index TICK true|false
python manage_stocks.py apply-json --file staged.json
```

Sub-commands: `list`, `add`, `set_status`, `set_index`, `apply-json`, and more.

## Outputs

- `monitored_stocks.parquet` — mutated in place

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [index_registry.md](index_registry.md) — resolves these flags to indexes
- [build_index.md](build_index.md) / [build_defensive_index.md](build_defensive_index.md) / [build_growth_tech_index.md](build_growth_tech_index.md)
- [inclusion_criteria.md](inclusion_criteria.md)
- [dual_screen_analysis.md](dual_screen_analysis.md) — `add` external candidates

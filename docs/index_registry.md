# index_registry.py

Discover and resolve portfolio / screen indexes from the data files. This is a
**library**, not a runnable script (imported by `cli_common` and most analytics
programs for `--index` resolution).

## Why it exists (rationale)

"Index" means different things in different scripts (a membership column, the
holdings, a sector EW series). This module centralizes the discovery and
canonical naming so `--index portfolio`, `--index growth`, `--universe all` all
resolve to the same concrete ticker lists everywhere.

## Canonical names / aliases

| Name | Source |
|------|--------|
| `fertilizer` | `index_member` |
| `defensive` | `defensive_value_index` |
| `growth` | `growth_tech_index` (alias `growth_tech`) |
| `dual` | `dual_pass_member` |
| `portfolio` | holdings / in_portfolio / trades |
| `sectors` | `SECT_*` sector EW series |
| `all` | every index available in the data |

## Key functions

- `discover_membership_indexes(stocks)` → name→column map
- `available_indexes(include_empty=False)` → list of valid index names
- `canonicalize(name)` → canonical name
- `parse_indexes(raw)` → resolved list (handles `all`, commas, aliases)
- `tickers_for_index(name)` → ticker list

## Outputs

None (library).

## Related programs

- [cli_common.md](cli_common.md) — uses this for `--index` resolution
- [manage_stocks.md](manage_stocks.md) — sets the membership columns
- Any program doc noting "flags via cli_common"

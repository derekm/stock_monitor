# cli_common.py

Shared CLI conventions and argparse helpers for stock_monitor programs. This is a
**library**, not a runnable script (imported by most programs for flag parsing
and ticker resolution).

## Why it exists (rationale)

Dozens of scripts need the same flags (`--index`, `--ticker`, `--sector`,
`--save`, `--window`, `--horizon`, `--freq`) and the same ticker/index
resolution. Centralizing it keeps every program's interface consistent and
avoids copy-paste drift.

## Standard flags

| Flag | Meaning |
|------|---------|
| `--index` | Index name(s), comma-separated or repeatable. **`all`** = every available index. |
| `--universe` | Hidden **alias** of `--index` (backward compatible). |
| `--ticker` | Comma-separated tickers (overrides `--index`). |
| `--tickers` | Hidden alias of `--ticker`. |
| `--sector` | Sector name or `SECT_` slug |
| `--save` | Write outputs |
| `--window` | Lookback trading days |
| `--horizon` | Forecast horizon |
| `--freq` | `D` / `W` / `M` |

Resolution order: `--ticker` → `--sector` → `--index`/`--universe` → program default.

## Key functions

- `build_parser(...)` — builds a parser with the standard flags.
- `add_index_args`, `add_ticker_args`, `add_sector_arg`, `add_save_arg`,
  `add_window_arg`, `add_horizon_arg`, `add_freq_arg` — granular adders.
- `resolve_tickers_from_args`, `resolve_index_names_from_args` — turn CLI args
  into concrete ticker lists (via `index_registry`).
- `add_window_arg` — adds `--window`.

## Outputs

None (library).

## Related programs

- [index_registry.md](index_registry.md) — index discovery used by resolution
- [data_access.md](data_access.md) — parquet loaders
- Every program doc that notes "flags via cli_common"

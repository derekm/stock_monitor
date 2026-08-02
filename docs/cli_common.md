# CLI conventions (`cli_common.py`)

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

## Resolution order

`--ticker` → `--sector` → `--index` / `--universe` → program default

## Libraries

- `index_registry.py` — discover indexes from data files
- `cli_common.py` — argparse helpers + `resolve_tickers_from_args`
- `data_access.py` — shared parquet loaders

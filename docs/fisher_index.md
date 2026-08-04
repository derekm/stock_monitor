# fisher_index.py

Chained Laspeyres / Paasche / Fisher price & quantity indexes from daily
close (price) and volume (quantity).

## Why it exists (rationale)

The Fisher index is the gold-standard chain index: it squares the bias of
pure Laspeyres (overstates) and Paasche (understates) by geometric-averaging
them. Used to track a real quantity-weighted basket level for the fertilizer /
ag-input universe (and any ticker set), with proper rebasing.

## Formulas (link t-1 → t)

$$L_P = \frac{\sum p_t q_{t-1}}{\sum p_{t-1} q_{t-1}},\quad
P_P = \frac{\sum p_t q_t}{\sum p_{t-1} q_t},\quad
F_P = \sqrt{L_P P_P}$$

(analogous for quantity $L_Q, P_Q, F_Q$).

## Usage

```bash
python fisher_index.py --universe portfolio --save
python fisher_index.py --universe all --ref-date 2020-01-01 --years 5
```

Flags (via `cli_common` + own): `--universe/--index`, `--ticker`, `--freq`
(D/W/M), `--save`, `--ref-date` (rebase to 100), `--years` (tail window).

## Outputs

- `fisher_indexes.csv` — index levels (Laspeyres / Paasche / Fisher, price & quantity)
- `fisher_indexes.parquet` — same, parquet

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [run_fisher_duckdb.md](run_fisher_duckdb.md) — DuckDB reimplementation / S&P reconciliation
- [fisher_sector_baskets.md](fisher_sector_baskets.md) — sector baskets
- [build_index.md](build_index.md) / [backfill_historical.md](backfill_historical.md)

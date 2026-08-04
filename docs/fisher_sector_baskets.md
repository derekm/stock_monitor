# fisher_sector_baskets.py

Fisher-style price indexes for sector baskets inside an index sleeve.

## Why it exists (rationale)

Extends `fisher_index.py` to sector baskets: builds equal-weight sector price
levels (by GICS sector for our sleeves, by `sp500_sector` for the S&P 500) and
exports a long panel for the dashboard's Fisher tab — so you can see
quantity-weighted sector drift, not just the headline index.

## Usage

```bash
python fisher_sector_baskets.py --index sp500 --save
python fisher_sector_baskets.py --index all --save
```

Flags: `--index` (sp500 / defensive / growth / fertilizer / portfolio / all),
`--lookback` (default 756 trading days), `--save`.

## Outputs

- `fisher_sector_baskets.csv` — basket levels over time
- `fisher_sector_baskets_latest.csv` — latest levels per basket

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [fisher_index.md](fisher_index.md) — base Fisher indexes
- [run_fisher_duckdb.md](run_fisher_duckdb.md)
- [sp_universe_tracking.md](sp_universe_tracking.md) / [parse_sp500.md](parse_sp500.md)

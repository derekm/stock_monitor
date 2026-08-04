# build_index.py

Construct a simple equal-weight Fertilizer / Ag-Inputs index from the active
`index_member` stocks and latest prices.

## Why it exists (rationale)

The core thematic index: tracks a basket of fertilizer / ag-input names the
personal book is built around. It is the benchmark the portfolio is measured
against and feeds index backtests.

## Usage

```bash
python build_index.py
```

Flags: none (reads `monitored_stocks.parquet` where `index_member=True`). Writes
`fertilizer_index.parquet` and prints the current snapshot and prior-day
performance.

## Outputs

- `fertilizer_index.parquet` — daily equal-weight index level + component returns

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [build_defensive_index.md](build_defensive_index.md) / [build_growth_tech_index.md](build_growth_tech_index.md)
- [live_index_backtest.md](live_index_backtest.md) — backtests the index
- [fisher_index.md](fisher_index.md) / [run_fisher_duckdb.md](run_fisher_duckdb.md)
- [manage_stocks.md](manage_stocks.md) — sets `index_member`

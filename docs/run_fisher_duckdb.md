# run_fisher_duckdb.py

Chained Fisher indexes computed **in DuckDB** (same logic as dashboard DuckDB-Wasm).

## SQL pipeline
1. Ffill quantity by ticker  
2. Adjacent date pairs (`lag`)  
3. Basket sums → Laspeyres/Paasche links → Fisher √(L·P)  
4. Chain with `100 * exp(sum(ln(link)) OVER (ORDER BY date …))`

```bash
python run_fisher_duckdb.py --universe portfolio --save
python run_fisher_duckdb.py --universe fertilizer --save
python run_fisher_duckdb.py --sector Materials --save
python run_fisher_duckdb.py --universe defensive --freq W --save
```

Outputs: `fisher_indexes_duckdb.csv` / `.parquet`  
SQL source: `fisher_index_duckdb.sql` (also `CORE_SQL` in the runner).

Dashboard **Fisher Indexes** tab can recompute the same SQL via DuckDB-Wasm or load these files.

## Related programs

- [docs/fisher_index.md](fisher_index.md)
- [docs/update_prices.md](update_prices.md)
- [docs/build_index.md](build_index.md)
- [docs/fisher_sector_baskets.md](fisher_sector_baskets.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

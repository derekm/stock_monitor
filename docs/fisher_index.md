# fisher_index.py

Chained **Laspeyres / Paasche / Fisher** price & quantity indexes in pure Python.

- **p** = close, **q** = volume (ffilled when sparse)
- Nominal: `fisher_p * fisher_q` and `sqrt(fisher_p * fisher_q)`
- Base level 100 on first date; levels = cumulative product of links

```bash
python fisher_index.py --universe portfolio --save
python fisher_index.py --sector Materials --save
python fisher_index.py --tickers MOS,CF,SHEL --freq W --save
```

Prefer **[run_fisher_duckdb.md](run_fisher_duckdb.md)** for the DuckDB implementation used as system of record.

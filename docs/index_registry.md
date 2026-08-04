# index_registry.py

Discover and resolve indexes from data files (`monitored_stocks.parquet`, holdings, sectors).

```bash
python index_registry.py
python forecast_granite.py forecast --index all --horizon 5
python forecast_granite.py forecast --index portfolio,growth
```

API: `available_indexes()`, `parse_indexes()`, `tickers_for_index()`, `ticker_index_map()`, `index_help_text()`.

Alias **`all`** expands to every index present in the data.

## Related programs

- [docs/build_index.md](build_index.md)
- [docs/build_growth_tech_index.md](build_growth_tech_index.md)
- [docs/build_defensive_index.md](build_defensive_index.md)
- [docs/cli_common.md](cli_common.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

# index_registry.py

Discover and resolve indexes from data files (`monitored_stocks.parquet`, holdings, sectors).

```bash
python index_registry.py
python forecast_granite.py forecast --index all --horizon 5
python forecast_granite.py forecast --index portfolio,growth
```

API: `available_indexes()`, `parse_indexes()`, `tickers_for_index()`, `ticker_index_map()`, `index_help_text()`.

Alias **`all`** expands to every index present in the data.

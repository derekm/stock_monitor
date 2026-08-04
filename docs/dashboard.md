# index.html — Portfolio Decision Dashboard

Single-page app: embedded analysis tables, **DuckDB-Wasm SQL Lab**, Chart.js, Fisher indexes, value screens.

## Serve
```bash
# from the stock_monitor repo root — starts all backend services + static site
./start_dashboard.sh
# open http://127.0.0.1:8765/index.html
```

## Tabs
| Tab | Content |
|-----|---------|
| Decisions | Inclusion memos, value trifecta, portfolio valuation |
| Portfolio | Holdings, home-sector correlations |
| Value Screens | Low EV/EBITDA, low P/B, fundamentals |
| Sectors | Corr summary chart, stability, HMM |
| Forecasts | Granite/fallback % change |
| Anomalies | TSPulse-ready flags |
| Index Backtest | Fertilizer / Defensive / Personal |
| Fisher Indexes | DuckDB-Wasm / JS / precomputed chained indexes + Chart.js |
| SQL Lab | Query builder + decision templates |
| CSV Catalog | SQL that reproduces analysis CSV artifacts |

## Data
`dashboard_data/data.json` — generated from parquet/csv; refresh after major pipeline runs.

## Fisher in-browser
**Compute in DuckDB-Wasm** runs `CORE_SQL` from `run_fisher_duckdb.py` against `price_qty_panel` and selected tickers; Chart.js plots Laspeyres, Paasche, Fisher, and nominal paths.

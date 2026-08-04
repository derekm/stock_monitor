# update_prices.py

Append daily open/close (and OHLC) to `daily_prices.parquet`.

## Why it exists (rationale)

`daily_prices.parquet` is the spine every analytic reads. This is the primary
way to extend it: fetch the last few days via yfinance (when network is
available), or enter/import manually (CSV / `manual`). It merges on
(date, ticker), keeping the newest source on conflict — the incremental
counterpart to `backfill_historical`.

## Usage

```bash
python update_prices.py fetch --days 5          # yfinance (needs network)
python update_prices.py manual --ticker CF --date 2026-07-28 --close 130.5
python update_prices.py import --csv new_prices.csv
```

Sub-commands: `fetch`, `manual`, `import`. `fetch` flags: `--days` (default 5),
`--tickers`. Reads/writes `daily_prices.parquet`, `monitored_stocks.parquet`.

## Outputs

- `daily_prices.parquet` — appended/updated rows

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [backfill_historical.md](backfill_historical.md) — bulk history
- [run_daily_automation.md](run_daily_automation.md) — refresh step
- [data_integrity.md](data_integrity.md) — post-update checks
- [ttm_features.md](ttm_features.md) / [fisher_index.md](fisher_index.md)

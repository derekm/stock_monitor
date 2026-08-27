# backfill_constituents.py

Fill **real, point-in-time** multi-snapshot fundamentals + price history for S&P
500 constituents missing from our store, using yfinance. This is the "fill in
real actuals" step for the S&P-tracking subsystem.

## Why it exists (rationale)

`sp_universe_tracking.py` / `sp_index_methodology.py` need genuine fundamentals
history for all 503 constituents, but our `fundamentals.parquet` only has the
personal book. This script fetches real quarterly financials + 5y daily prices
per missing constituent, derives the canonical quality metrics per quarter-end,
and writes them in the **same schema** as `fundamentals.parquet` /
`daily_prices/` (so the rest of the stack can use them unchanged).

## Usage

```bash
python backfill_constituents.py run            # backfill all missing constituents
python backfill_constituents.py run --limit 20 # smoke test on 20 tickers
python backfill_constituents.py merge          # union staging into the real files
python backfill_constituents.py status          # show progress
```

Flags: `--limit N` (smoke test), `--sleep F` (rate-limit pause, default 0.4s).

## Outputs

- `fundamentals_yfinance.parquet` — staging fundamentals (derived roe, roic,
  debt_to_equity, interest_coverage, ev_ebitda, mktcap_to_assets, pb_ratio per
  quarter-end; `source='yfinance'`)
- `daily_prices_yfinance.parquet` — staging 5y daily price/volume
- `backfill_progress.json` — resume marker (run is resume-safe)
- `sp500_constituents.parquet` — the authoritative constituent list it reads from
- After `merge`: appends into `fundamentals.parquet` and `daily_prices/`

(Schema families: base_table / summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [parse_sp500.md](parse_sp500.md) — builds `sp500_constituents.parquet`
- [sp_universe_tracking.md](sp_universe_tracking.md)
- [sp_index_methodology.md](sp_index_methodology.md)
- [update_fundamentals.md](update_fundamentals.md) / [update_prices.md](update_prices.md) — same schema targets

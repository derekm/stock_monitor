# backfill_edgar.py

Real decades-long point-in-time fundamentals from SEC EDGAR XBRL companyfacts (v2 + HTML 10-Q).

## Why it exists (rationale)

`update_fundamentals.py fetch-history` (yfinance) tops out at ~5–8 quarters of statement history — too shallow for honest long-window factor backtests. EDGAR's XBRL companyfacts JSON (public, no key, UA header + 10 req/s limit) goes back to ~2009 for most names. This writer pulls it and computes the same as-of-quarter-end metrics the yfinance writer does, so the point-in-time consumers (`cross_section.py`, `signal_aggregator.py`, `peer_analytics.py`) see ~15 years instead of 2.

## Method

Per ticker (via SEC ticker→CIK map):
- Income rows: keep only single-quarter entries (CYyyyyQn frames), one per quarter (last filed wins). TTM = rolling 4-quarter sum.
- Balance rows: `end`-dated point-in-time values.
- Metrics at each quarter end:
  - ROE = TTM NetIncomeLoss / StockholdersEquity
  - ROIC = TTM NOPAT / invested capital (NOPAT = OperatingIncomeLoss × 0.75; invested = InvestedCapital, fallback debt + equity)
  - D/E = TotalDebt / StockholdersEquity
  - EV/EBITDA = (mktcap + debt − cash) / TTM EBITDA (EBITDA = OI + D&A, fallback OI-only where D&A is unreported)
  - P/B = mktcap / equity; MktCap/Assets = mktcap / assets
  - interest_coverage = TTM OI / TTM interest expense
- Market cap = adj_close price × shares at the quarter end (from `daily_prices.parquet`; shares from EDGAR, last value ≤ qend).

Rows are `source=edgar_v2` or `source=html_10q` and displace lower-priority sources (`yfinance_history`, `polygon_financials`, `fundamentals_history_backfill`) for the same (ticker, period) — EDGAR is deeper and point-in-time, so it wins.

## Key Features (2026-08)

- **Incremental flush**: `FLUSH_EVERY=40` tickers, writes to `fundamentals.parquet` mid-run — crash-safe.
- **Protected sources**: Never overwrites `edgar_v2` or `html_10q` rows with lower-priority data.
- **_old column restore**: After pandas merge with suffixes, copies `_old` columns back to bare names before applying protected overwrite logic.
- **Future date filter**: Drops incoming rows with `as_of_date > today` before merge.
- **Prior estimates preserved**: Future-quarter EDGAR estimates (from XBRL frames like CY2025Q1, CY2025Q2 etc.) are stored as `prior_estimate_<metric>` columns on the most recent actual quarter. This enables tracking how well companies meet/beat analyst expectations and measuring health toward targets. `--migrate-future-estimates` applies the same fold to pre-existing future-dated rows written before the split existed (backs up first, then removes them from the time series).
- **Test-row purge**: `--purge-test-tickers` removes `TEST*` rows left behind by merge smoke tests.
- **FILL_COLS policy**: Only fills missing columns (`FILL_COLS='c'`), never overwrites existing values.
- **Quarantine**: 404s and entity-name mismatches are quarantined, not failed.
- **Local CIK map**: `cik_ticker_map.json` (10,390 entries) wins over live SEC lookup.
- **Numeric coercion**: All numeric columns coerced to float64 to avoid pyarrow overflow.

## Usage

```bash
python backfill_edgar.py --dry-run          # CIK coverage report only
python backfill_edgar.py --max-tickers 150
python backfill_edgar.py --tickers AAPL,MSFT,JPM
python backfill_edgar.py --quarantine       # full universe with quarantine
python backfill_edgar.py --resume           # resume from checkpoint
python backfill_edgar.py --purge-test-tickers        # drop TEST* smoke-test rows
python backfill_edgar.py --migrate-future-estimates  # fold legacy future rows into prior_estimate_*
```

## Outputs

- Writes into `fundamentals.parquet` (base table, same schema family as `update_fundamentals.py`). No new output file.
- Checkpoint: `backfill_checkpoints/fund_backfill.json` (versioned schema v2)

## Notes / limitations

- ETFs/funds have no XBRL statements (404 / empty) — skipped with `!!`.
- ~8,600/9,955 universe tickers have usable EDGAR data; coverage varies (IPO'd names like RKLB ~19 quarters).
- NOPAT uses a 25% tax proxy; D&A is sparsely reported so EBITDA often falls back to OI-only.
- Respects SEC rate limits (0.12s sleep between requests).
- HTML 10-Q parsing preferred over incomplete XBRL for recent quarters.
- Universe = `daily_prices.parquet` (not `monitored_stocks.parquet`).

## Related programs

- `update_fundamentals.py` — yfinance shallow alternative (fetch-history)
- `cross_section.py` / `signal_aggregator.py` / `peer_analytics.py` — point-in-time factor consumers that benefit from the deeper history
- `acquisition_backfill.py` — delegates to this for new ticker backfill
- `resumable_job.py` — checkpoint framework
- `edgar_companyfacts_v2.md`, `edgar_html_10q.md` — extractor details
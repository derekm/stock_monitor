# backfill_edgar.py

Real decades-long point-in-time fundamentals from SEC EDGAR XBRL companyfacts.

## Why it exists (rationale)

`update_fundamentals.py fetch-history` (yfinance) tops out at ~5–8 quarters of
statement history — too shallow for honest long-window factor backtests.
EDGAR's XBRL companyfacts JSON (public, no key, UA header + 10 req/s limit)
goes back to ~2009 for most names. This writer pulls it and computes the same
as-of-quarter-end metrics the yfinance writer does, so the point-in-time
consumers (`cross_section.py`, `signal_aggregator.py`) see ~15 years instead
of 2.

## Method

Per ticker (via SEC ticker→CIK map):
- Income rows: keep only single-quarter entries (CYyyyyQn frames), one per
  quarter (last filed wins). TTM = rolling 4-quarter sum.
- Balance rows: `end`-dated point-in-time values.
- Metrics at each quarter end:
  - ROE = TTM NetIncomeLoss / StockholdersEquity
  - ROIC = TTM NOPAT / invested capital (NOPAT = OperatingIncomeLoss × 0.75;
    invested = InvestedCapital, fallback debt + equity)
  - D/E = TotalDebt / StockholdersEquity
  - EV/EBITDA = (mktcap + debt − cash) / TTM EBITDA (EBITDA = OI + D&A,
    fallback OI-only where D&A is unreported)
  - P/B = mktcap / equity; MktCap/Assets = mktcap / assets
  - interest_coverage = TTM OI / TTM interest expense
- Market cap = adj_close price × shares at the quarter end (from
  `daily_prices.parquet`; shares from EDGAR, last value ≤ qend).

Rows are `source=edgar` and displace BOTH the synthetic
`fundamentals_history_backfill` rows and the shallow `yfinance_history` rows
for the same tickers — EDGAR is deeper and point-in-time, so it wins.

## Usage

```bash
python backfill_edgar.py --dry-run          # CIK coverage report only
python backfill_edgar.py --max-tickers 150
python backfill_edgar.py --tickers AAPL,MSFT,JPM
python backfill_edgar.py                    # all monitored tickers
```

## Outputs

- Writes into `fundamentals.parquet` (base table, same schema family as
  `update_fundamentals.py`). No new output file.

## Notes / limitations

- ETFs/funds have no XBRL statements (404 / empty) — skipped with `!!`.
- ~111/142 monitored tickers have usable EDGAR data; coverage varies
  (IPO'd names like RKLB ~19 quarters, XOM shows 2 due to CIK mismatch —
  re-run after verifying the CIK map entry).
- NOPAT uses a 25% tax proxy; D&A is sparsely reported so EBITDA often falls
  back to OI-only.
- Respects SEC rate limits (0.12s sleep between requests).

## Related programs

- `update_fundamentals.py` — yfinance shallow alternative (fetch-history)
- `cross_section.py` / `signal_aggregator.py` / `peer_analytics.py` —
  point-in-time factor consumers that benefit from the deeper history
- `cv_utils.py` — the OOS discipline this data supports

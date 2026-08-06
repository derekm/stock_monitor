# update_earnings.py

Maintain the canonical earnings calendar: upcoming + historical earnings
per ticker.

## Why it exists (rationale)

Nothing in the repo previously had earnings dates or surprises — the
`earnings_catalyst.py` filter needs them. This writer populates
`earnings_calendar.parquet` via yfinance `get_earnings_dates()` (EPS estimate,
reported EPS, surprise % per quarter), following the same writer pattern as
`update_fundamentals.py` (dedup on ticker+key, keep-last).

## Usage

```bash
python update_earnings.py fetch                # all monitored tickers, capped at 60
python update_earnings.py fetch --ticker AAPL,MSFT
python update_earnings.py fetch --days 12 --max-tickers 100
python update_earnings.py show --ticker AAPL
python update_earnings.py show --days 12       # last 12 months
```

Fetch is per-ticker try/except so one bad symbol (e.g. an ETF with no earnings
dates) can't kill the batch — it logs `!! TICKER: reason` and continues.

## Outputs

- `earnings_calendar.parquet` — SCHEMAS family `earnings`:
  `ticker`, `earnings_date` (DATE), `eps_estimate`, `reported_eps`,
  `surprise_pct`, `source`, `last_updated`

## Related programs

- `earnings_catalyst.py` — consumer (drift buckets, pre-earnings momentum, IV flag)
- `update_fundamentals.py` — same writer pattern, separate table

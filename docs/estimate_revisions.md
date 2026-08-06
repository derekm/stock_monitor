# estimate_revisions.py

Consensus EPS-estimate and price-target snapshots with revision tracking.

## Why it exists (rationale)

Closes the "estimate revisions" TODO. Analyst consensus changes are a leading
fundamental signal. yfinance exposes current consensus but no history — so
this script SNAPSHOTS it into `estimate_revisions.parquet` on each run
(append), and the revision columns compare the latest snapshot against the
previous one.

## Method

- `earnings_estimate` (avg per period: 0q, +1q, 0y, +1y) and
  `analyst_price_targets` (mean) per ticker.
- Append-only parquet keyed by (snapshot_date, ticker, period).
- `mean_eps_rev_pct` / `mean_pt_rev_pct` = pct change vs the previous
  snapshot for the same (ticker, period).

First run seeds the baseline (revisions NaN); subsequent daily runs produce
revisions. Run at least twice with a gap to see meaningful data.

## Usage

```bash
python estimate_revisions.py --save
python estimate_revisions.py --save --tickers AAPL,MSFT
```

## Outputs

- `estimate_revisions.parquet` — long table.

## Related programs

- `earnings_catalyst.py` — earnings surprise/PEAD signals
- `update_earnings.py` — the earnings calendar this complements

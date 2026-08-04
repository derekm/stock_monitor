# portfolio_report.py

Snapshot of holdings, cost basis, and P&L from `trades.parquet` + latest prices.

## Why it exists (rationale)

The personal book is the thing being managed. This prints a current holdings
snapshot (shares, cost basis, market value, unrealized P&L) either from the stored
`portfolio_holdings.parquet` or recomputed fresh from the trade log + latest
closes. It is the human-readable view of where the fund stands.

## Usage

```bash
python portfolio_report.py
python portfolio_report.py --refresh    # rebuild holdings from trades + current prices
```

Flags: `--refresh` (recompute `portfolio_holdings.parquet` from trades + prices).
Reads `trades.parquet`, `daily_prices.parquet`, `monitored_stocks.parquet`,
`fundamentals.parquet`.

## Outputs

- `portfolio_holdings.parquet` — (re)built when `--refresh` is passed
- Prints the snapshot to stdout

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [run_daily_automation.md](run_daily_automation.md) — refresh step
- [vol_target.md](vol_target.md) — sizing against these holdings
- [buy_candidates.md](buy_candidates.md) / [preferred_metrics.md](preferred_metrics.md)

# fundamentals_history.py — real point-in-time history

`update_fundamentals.py fetch-history` replaces the synthetic mean-reverting
backfill (`source=fundamentals_history_backfill`) with **real quarterly
statements** from yfinance.

## What changed

- **Before**: `fundamentals_history.py backfill` invented history by adding
  mean-reverting noise around the latest row — fine for screen backtest
  plumbing, but the "history" was synthetic and could not support honest OOS
  factor work.
- **After**: `update_fundamentals.py fetch-history --max-tickers N` pulls the
  quarterly income statement + balance sheet per ticker and computes
  as-of-quarter-end: ROE (TTM NI / equity), ROIC (TTM NOPAT / invested
  capital), D/E, EV/EBITDA ((mktcap + debt − cash) / TTM EBITDA), P/B
  (mktcap / equity), MktCap/Assets. Market cap = adj_close price × shares at
  the quarter end. Rows are `source=yfinance_history` and displace synthetic
  backfill rows for the same tickers.

## Usage

```bash
python update_fundamentals.py fetch-history --max-tickers 150
python update_fundamentals.py fetch-history --tickers AAPL,MSFT,JPM
```

## Notes / limitations

- ETFs and funds have no statements and are skipped (`!! TICKER: no
  statements`).
- NOPAT uses a ~25% effective-tax proxy (`OperatingIncome × 0.75`); ROIC is
  therefore approximate, not a tax-computed figure.
- yfinance statement history depth varies (5–8 quarters per name); it does
  not reach back to 1962 like prices do. Deeper history would require SEC
  EDGAR (see remaining-work notes).
- The `backfill` subcommand in `fundamentals_history.py` still exists for
  screen-backtest plumbing; prefer real history where it exists.

## Related programs

- `update_fundamentals.py` — the writer (fetch-history subcommand)
- `cross_section.py` / `signal_aggregator.py` — point-in-time factor consumers
- `fundamentals_history.py` — snapshot / screen backtest on the dated rows

# shadow_book.py

Paper-trade the fund-in-construction: replay buy_candidates target weights
against realized prices, with FIFO tax lots and kill switches.

## Why it exists (rationale)

Closes the "paper trading / shadow mode", "kill switches", and "tax-lot
awareness" TODOs. The Robinhood portfolio is real; a shadow book lets a
target-weight signal prove itself against realized prices before capital
moves, and enforces the kill-switch layer (max drawdown, vol spike) that
would halt rebalancing.

## Method

- Target weights from `buy_candidates.csv`: BUY/ACCUMULATE → hold,
  AVOID → none, else 0.5 weight; equal-weight notional split.
- Fills at the NEXT trading day's close after the signal date (no lookahead).
- FIFO tax lots: sells consume the oldest lot; realized PnL approximated from
  cost basis; open lots written out.
- Kill switches: 25% drawdown from high-water mark, 21d annualized vol > 60%,
  insufficient price history. When tripped the book stops trading (state
  recorded per day).

## Usage

```bash
python shadow_book.py --save --days 504
python shadow_book.py --save --cash 250000
```

## Outputs

- `shadow_book.csv` — per-day (equity, cash, drawdown, n_holdings,
  kill_switch) + ann_ret/sharpe
- `shadow_lots.csv` — open FIFO lots

## Related programs

- `buy_candidates.py` — the target-weight source
- `portfolio_optimization.py` — the real-weight allocator this shadows
- `perf_metrics.py` — the metric set for evaluating the book

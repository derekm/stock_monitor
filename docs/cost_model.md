# cost_model.py

Shared trading-cost assumptions for backtests.

## Why it exists (rationale)

Every backtest used to be cost-free — a monthly-rebalanced L/S portfolio and a
pair-trading book have very different cost profiles, and an honest OOS number
must be NET. This module is threaded through `cross_section.py` (per-day
turnover cost) and `pair_engine.py` (per-trade round trip + short borrow).

Defaults (conservative retail):
- `ROUND_TRIP_BPS = 10` — commission + spread + slippage, per side
- `BORROW_BPS = 25` — annualized on short notional (hard-to-borrow names)

## Usage

```python
from cost_model import apply_costs_to_daily, apply_costs_to_trades

net_daily = apply_costs_to_daily(rets, turnover_frac=1.0/21.0)
net_trades = apply_costs_to_trades(trades, pnl_col="hedged_pnl")
```

## Outputs

None (library only). `apply_costs_to_trades` adds `net_<pnl_col>` and `cost`
columns to the trades frame.

## Related programs

- `cross_section.py` — per-day cost on L/S + baseline returns
- `pair_engine.py` — per-trade cost; pair stats computed on `net_hedged_pnl`
- `cv_utils.py` — the OOS stats these net returns feed

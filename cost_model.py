#!/usr/bin/env python3
"""
cost_model.py — Shared trading-cost assumptions for backtests.

Threaded through cross_section.py and pair_engine.py so every reported
return is NET of costs. Defaults are conservative for a retail account:

  round_trip_bps   : 10 bps per side (commission + spread + slippage)
  borrow_bps       : 25 bps annualized on short notional (hard-to-borrow retail)
  min_hold_days    : 1 (no minimum)

Usage:
  from cost_model import apply_costs_to_daily, apply_costs_to_trades
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROUND_TRIP_BPS = 10.0
BORROW_BPS = 25.0


def apply_costs_to_daily(
    rets: pd.DataFrame,
    turnover_frac: float,
    round_trip_bps: float = ROUND_TRIP_BPS,
) -> pd.DataFrame:
    """Subtract per-day trading cost from a daily-return portfolio frame.

    ``turnover_frac`` = fraction of notional traded per day (1.0 for full
    monthly rebalance spread over ~21 days; pass the per-day average).
    Cost per day = turnover_frac * round_trip_bps / 1e4.
    """
    cost_per_day = turnover_frac * round_trip_bps / 1e4
    out = rets.copy()
    for col in rets.columns:
        out[col] = rets[col] - cost_per_day
    return out


def apply_costs_to_trades(
    trades: pd.DataFrame,
    pnl_col: str = "hedged_pnl",
    round_trip_bps: float = ROUND_TRIP_BPS,
    borrow_bps: float = BORROW_BPS,
    notional: float = 1.0,
) -> pd.DataFrame:
    """Subtract costs from per-trade PnL.

    Trade cost = round_trip_bps/1e4 * notional (both legs), plus borrow cost
    accrued over bars_held: borrow_bps/1e4 * notional * bars_held/252.
    """
    out = trades.copy()
    if pnl_col not in out.columns or len(out) == 0:
        return out
    trade_cost = round_trip_bps / 1e4 * notional
    if "bars_held" in out.columns:
        borrow = borrow_bps / 1e4 * notional * out["bars_held"].fillna(0) / 252.0
    else:
        borrow = 0.0
    out["net_" + pnl_col] = out[pnl_col] - trade_cost - borrow
    out["cost"] = trade_cost + borrow
    return out


if __name__ == "__main__":
    # smoke
    r = pd.DataFrame({"long_short": [0.01, -0.005, 0.008]})
    net = apply_costs_to_daily(r, turnover_frac=1.0 / 21.0)
    print(net)
    t = pd.DataFrame({"hedged_pnl": [0.02, -0.01], "bars_held": [10, 40]})
    print(apply_costs_to_trades(t))

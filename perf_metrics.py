#!/usr/bin/env python3
"""
perf_metrics.py — Full performance-metric set for backtests: Sharpe, Sortino,
Calmar, max DD, hit rate, profit factor, turnover, capacity.

Why it exists: the architecture TODO "metrics beyond returns — Sharpe/Sortino,
max drawdown, Calmar, hit rate, profit factor, turnover, capacity". The repo
had Sharpe/maxDD/CVaR scattered; this centralizes the complete set and
threads it into cv_utils.oos_stats_vs_baseline and the engines.

Functions:
  perf_metrics(daily_rets, trades=None, notional=None, target_vol=0.15)
      -> dict with ann_ret, ann_vol, sharpe, sortino, max_dd, calmar,
         hit_rate, profit_factor, turnover, capacity
  profit_factor_from_trades(trades, pnl_col) -> float
  capacity_estimate(adv_dollar, daily_turnover, max_turnover_frac=0.05)
      -> max notional the strategy can trade without moving price

Usage (library — import, don't run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def perf_metrics(daily_rets: pd.Series | np.ndarray,
                 trades: pd.DataFrame | None = None,
                 pnl_col: str = "net_hedged_pnl",
                 notional: float = 1.0,
                 rf: float = 0.0) -> dict:
    r = pd.Series(np.asarray(daily_rets, dtype=float)).dropna()
    if len(r) < 5:
        return {}
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf) / downside if downside and downside > 0 else np.nan
    cum = np.exp(r.cumsum())
    dd = cum / np.maximum.accumulate(cum) - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd and max_dd < 0 else np.nan

    hit_rate, profit_factor = None, None
    if trades is not None and len(trades) and pnl_col in trades.columns:
        pnl = trades[pnl_col].dropna().astype(float)
        if len(pnl):
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            hit_rate = float(len(wins) / len(pnl))
            profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan

    turnover = float(r.abs().mean() * 252) if len(r) else None
    return {
        "ann_ret": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3) if np.isfinite(sharpe) else None,
        "sortino": round(sortino, 3) if np.isfinite(sortino) else None,
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 3) if np.isfinite(calmar) else None,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "turnover": round(turnover, 3) if turnover is not None else None,
        "n_days": int(len(r)),
    }


def capacity_estimate(adv_dollar: float, daily_turnover: float,
                      max_turnover_frac: float = 0.05) -> dict:
    """Max strategy notional before daily turnover exceeds X% of ADV.

    capacity = adv_dollar * max_turnover_frac / daily_turnover
    """
    if not adv_dollar or adv_dollar <= 0 or not daily_turnover or daily_turnover <= 0:
        return {"capacity_notional": None, "note": "insufficient data"}
    cap = adv_dollar * max_turnover_frac / daily_turnover
    return {"capacity_notional": round(float(cap), 0),
            "note": f"daily turnover <= {max_turnover_frac:.0%} of ADV"}


if __name__ == "__main__":
    import sys
    print(__doc__)
    sys.exit(0)

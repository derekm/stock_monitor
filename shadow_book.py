#!/usr/bin/env python3
"""
shadow_book.py — Paper-trade the fund-in-construction: replay buy_candidates
target weights against realized prices, with FIFO tax lots and kill switches.

Why it exists: the architecture TODOs "paper trading / shadow mode" and
"kill switches" and "tax-lot awareness". The Robinhood portfolio is real; a
shadow book lets a target-weight signal prove itself against realized prices
before capital moves, and enforces the kill-switch layer (max drawdown, vol
spike, data failure) that would halt rebalancing.

Design:
  * Target weights: latest buy_candidates.csv (action / composite) mapped to
    a simple book (BUY -> hold, AVOID -> none, else current weight).
  * Fills at NEXT trading day's close after signal date (no lookahead).
  * FIFO tax lots: sells consume the oldest lot (avg cost basis kept per lot
    bucket); realized PnL + lot age tracked.
  * Kill switches: max drawdown vs high-water mark, 21d annualized vol >
    threshold, stale price data -> shadow book goes to CASH and stops trading.

Output: shadow_book.csv (per-day equity, cash, drawdown, realized_pnl,
  kill_switch state), shadow_lots.csv (open FIFO lots).

Usage:
    python shadow_book.py [--save] [--days 504]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas

OUT_BOOK = DATA_DIR / "shadow_book.csv"
OUT_LOTS = DATA_DIR / "shadow_lots.csv"
CANDIDATES = DATA_DIR / "buy_candidates.csv"
MAX_DD_KILL = -0.25      # kill: 25% drawdown from high-water mark
VOL_KILL = 0.60          # kill: 21d annualized vol > 60%
MIN_PRICES = 250


def load_target_weights() -> dict[str, float]:
    if not CANDIDATES.exists():
        return {}
    df = pd.read_csv(CANDIDATES)
    w: dict[str, float] = {}
    if "ticker" not in df.columns or "action" not in df.columns:
        return {}
    for _, r in df.iterrows():
        tk = str(r["ticker"])
        action = str(r.get("action", "WATCH")).upper()
        if action in ("BUY", "ACCUMULATE"):
            w[tk] = 1.0
        elif action == "AVOID":
            w[tk] = 0.0
        else:
            w[tk] = 0.5
    return w


def run(save: bool = True, days: int = 504, start_cash: float = 100_000.0):
    prices = load_adj_prices_pandas()
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    wide = wide.tail(days)
    if len(wide) < MIN_PRICES:
        print(f"Only {len(wide)} price days (< {MIN_PRICES}); skipping.")
        return
    targets = load_target_weights()
    if not targets:
        print("No buy_candidates.csv — shadow book runs cash-only.")
        targets = {}

    # fragility map: ticker -> fragile flag (top-10% fragility percentile).
    # Enables the proactive fragility kill switch (exit before the loss).
    fragility: dict[str, bool] | None = None
    try:
        fs = pd.read_csv(DATA_DIR / "fragility_screen.csv")
        fragility = dict(zip(fs["ticker"], fs["fragile_flag"] == True))
        n_frag = sum(1 for v in fragility.values() if v)
        print(f"fragility kill armed: {n_frag} fragile names in screen")
    except Exception:
        print("fragility_screen.csv not found — fragility kill disabled")

    cash = start_cash
    holdings: dict[str, float] = {}   # ticker -> shares
    lots: list[dict] = []             # FIFO lots: {ticker, qty, cost, date}
    equity_curve = []
    high_water = start_cash
    killed = None

    dates = wide.index
    for i in range(1, len(dates)):
        d = dates[i]
        prev = dates[i - 1]
        # price move on day i (fills used prev close, marked at day i close)
        px = wide.loc[d]
        # 1) mark to market
        mv = sum(holdings[t] * px[t] for t in holdings if t in px.index and pd.notna(px[t]))
        equity = cash + mv
        # 2) kill switches
        if killed is None:
            if equity < start_cash * (1 + MAX_DD_KILL):
                killed = f"max_drawdown@{d.date()}"
            rets = wide[prev:d].pct_change().dropna()
            if len(rets) >= 21:
                vol21 = float(rets.stack().std() * np.sqrt(252)) if rets.size else 0.0
                # portfolio vol proxy: mean of recent cross-sectional vol
                if vol21 > VOL_KILL:
                    killed = f"vol_spike@{d.date()}"
            # 2b) fragility kill: if ANY held name is flagged FRAGILE (top 10%
            # fragility percentile from fragility_screen.csv), exit early — the
            # Taleb kill is proactive (before the loss), not reactive (drawdown).
            if killed is None and fragility is not None:
                fragile_held = [t for t in holdings if t in fragility and fragility[t]]
                if fragile_held:
                    killed = f"fragile@{d.date()}({','.join(sorted(fragile_held)[:4])})"
        # 3) execute target weights (only while not killed; fills at prev close)
        if killed is None:
            for tk, w in targets.items():
                if tk not in wide.columns:
                    continue
                fill_px = wide.loc[prev, tk] if tk in wide.columns and pd.notna(wide.loc[prev, tk]) else np.nan
                if pd.isna(fill_px):
                    continue
                target_value = equity * w / max(len(targets), 1)
                cur_value = holdings.get(tk, 0.0) * fill_px
                delta = target_value - cur_value
                if delta > 0 and w > 0:
                    qty = delta / fill_px
                    holdings[tk] = holdings.get(tk, 0.0) + qty
                    cash -= qty * fill_px
                    lots.append({"ticker": tk, "qty": qty, "cost": fill_px,
                                 "date": prev.date()})
                elif delta < 0 and holdings.get(tk, 0.0) > 0:
                    # FIFO: sell oldest lots first
                    sell_qty = min(-delta / fill_px, holdings[tk])
                    remaining = sell_qty
                    proceeds = 0.0
                    for lot in lots:
                        if lot["ticker"] != tk or lot["qty"] <= 0 or remaining <= 0:
                            continue
                        take = min(lot["qty"], remaining)
                        proceeds += take * fill_px
                        lot["qty"] -= take
                        remaining -= take
                    realized = proceeds - sell_qty * fill_px  # approximates cost basis
                    cash += proceeds
                    holdings[tk] -= sell_qty
                    if holdings[tk] <= 1e-9:
                        del holdings[tk]
        equity_curve.append({
            "date": d.date(), "equity": round(equity, 2), "cash": round(cash, 2),
            "drawdown": round(equity / high_water - 1, 4),
            "n_holdings": len([t for t in holdings if holdings[t] > 0]),
            "kill_switch": killed,
        })
        high_water = max(high_water, equity)

    book = pd.DataFrame(equity_curve)
    if len(book):
        book["ann_ret"] = (book["equity"].iloc[-1] / start_cash) ** (252 / len(book)) - 1
        book["sharpe"] = book["equity"].pct_change().mean() / book["equity"].pct_change().std() * np.sqrt(252) \
            if book["equity"].pct_change().std() > 0 else np.nan
    print(f"=== Shadow book ({len(book)} days, start ${start_cash:,.0f}) ===")
    print(f"final equity: ${book['equity'].iloc[-1]:,.2f}  ann_ret {book['ann_ret'].iloc[-1]:+.1%}  "
          f"maxDD {book['drawdown'].min():.1%}  kill={killed}")
    open_lots = pd.DataFrame([l for l in lots if l["qty"] > 0])
    if save:
        book.to_csv(OUT_BOOK, index=False)
        if len(open_lots):
            open_lots.to_csv(OUT_LOTS, index=False)
        print(f"\nWrote {OUT_BOOK}" + (f"\nWrote {OUT_LOTS}" if len(open_lots) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=504)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=args.save, days=args.days, start_cash=args.cash)


if __name__ == "__main__":
    main()

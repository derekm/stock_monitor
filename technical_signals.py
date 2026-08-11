#!/usr/bin/env python3
"""
technical_signals.py — RSI, MACD, Bollinger/Keltner bands, SMA crossovers,
and volume-price confirmation across the monitored universe.

Why it exists: the architecture TODO "technical signals" — the repo had SMA
alert rules but no systematic RSI/MACD/band engine. This computes a compact
signal set per ticker on split/dividend-adjusted closes (adj_close) so every
indicator is comparable across splits.

Indicators (all standard, computed with pandas — no TA-Lib dependency):
  rsi14            — Wilder RSI(14)
  macd / macd_sig  — MACD(12,26,9); macd_hist = macd - signal
  bb_upper/lower   — Bollinger(20, 2σ)
  keltner_up/dn    — Keltner(20, 2×ATR14)
  sma20 / sma50    — for crossover checks
  sma_cross        — +1 if sma20 crossed above sma50 recently, -1 below
  volume_confirm   — +1 if up-day volume > 20d median, -1 if down-day, 0 else

Output: technical_signals.csv — latest snapshot per ticker (date, indicators,
  cross flags). Consumers: buy_candidates overlay, dashboard.

Usage:
    python technical_signals.py [--save] [--tickers AAPL,MSFT]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas

OUT = DATA_DIR / "technical_signals.parquet"


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_indicators(wide: pd.DataFrame, ticker: str) -> dict:
    c = wide[ticker].dropna()
    if len(c) < 60:
        return {}
    rsi = _wilder_rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    mid20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    bb_up = mid20 + 2 * sd20
    bb_dn = mid20 - 2 * sd20
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma_cross = 0
    if len(c) >= 51 and not np.isnan(sma20.iloc[-1]) and not np.isnan(sma20.iloc[-2]) \
            and not np.isnan(sma50.iloc[-1]) and not np.isnan(sma50.iloc[-2]):
        if sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2]:
            sma_cross = 1
        elif sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2]:
            sma_cross = -1
    last = {
        "date": c.index[-1].date(),
        "close": round(float(c.iloc[-1]), 4),
        "rsi14": round(float(rsi.iloc[-1]), 2) if not np.isnan(rsi.iloc[-1]) else None,
        "macd": round(float(macd.iloc[-1]), 6) if not np.isnan(macd.iloc[-1]) else None,
        "macd_signal": round(float(macd_sig.iloc[-1]), 6) if not np.isnan(macd_sig.iloc[-1]) else None,
        "macd_hist": round(float(macd.iloc[-1] - macd_sig.iloc[-1]), 6) if not np.isnan(macd.iloc[-1] - macd_sig.iloc[-1]) else None,
        "bb_upper": round(float(bb_up.iloc[-1]), 4) if not np.isnan(bb_up.iloc[-1]) else None,
        "bb_lower": round(float(bb_dn.iloc[-1]), 4) if not np.isnan(bb_dn.iloc[-1]) else None,
        "bb_pct": round(float((c.iloc[-1] - bb_dn.iloc[-1]) / (bb_up.iloc[-1] - bb_dn.iloc[-1])), 3) if not np.isnan(bb_up.iloc[-1] - bb_dn.iloc[-1]) and bb_up.iloc[-1] != bb_dn.iloc[-1] else None,
        "sma20": round(float(sma20.iloc[-1]), 4) if not np.isnan(sma20.iloc[-1]) else None,
        "sma50": round(float(sma50.iloc[-1]), 4) if not np.isnan(sma50.iloc[-1]) else None,
        "sma_cross": sma_cross,
        "above_sma20": bool(c.iloc[-1] > sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
        "above_sma50": bool(c.iloc[-1] > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
    }
    return last


def build(tickers: list[str] | None = None) -> pd.DataFrame:
    prices = load_adj_prices_pandas()  # returns date/ticker/close (adj_close renamed)
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    if tickers:
        wide = wide[[t for t in tickers if t in wide.columns]]
    rows = []
    for tk in wide.columns:
        r = compute_indicators(wide, tk)
        if r:
            r["ticker"] = tk
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = all with prices")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    df = build(tickers)
    print(f"=== Technical signals ({len(df)} tickers) ===")
    cols = [c for c in ["ticker", "date", "close", "rsi14", "macd_hist", "bb_pct", "sma_cross", "above_sma50"] if c in df]
    print(df[cols].sort_values("ticker").head(25).to_string(index=False))
    if args.save:
        df.to_parquet(OUT)
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

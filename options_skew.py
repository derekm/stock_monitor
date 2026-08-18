#!/usr/bin/env python3
"""
options_skew.py — Implied-vol skew and put/call volume ratios from the
yfinance options chain (monitored universe).

Why it exists: the architecture TODO "options IV skew, put/call ratios".
earnings_catalyst already fetches ATM IV; this adds the cross-sectional
skew signal (vol smile tilt) and the put/call volume ratio — the classic
fear/positioning indicators.

Computed per ticker (nearest-dated chain with sane IVs, ~30d out):
  atm_iv      — median IV of strikes within 5% of spot (sane IVs > 0.05)
  skew        — IV(strike = 0.9*spot) - IV(strike = 1.1*spot); positive =
                downside puts richer (fear). NaN when strikes unavailable.
  put_call_vol — total put volume / total call volume (nearest expiry)

Output: options_skew.csv — latest snapshot per ticker.

Usage:
    python options_skew.py [--save] [--max-tickers 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from analytics_common import DATA_DIR

OUT = DATA_DIR / "options_skew.parquet"


def ticker_options_metrics(t: str, target_days: int = 30) -> dict:
    """(atm_iv, skew, put_call_vol) for one ticker, or {} when no chain."""
    tk = yf.Ticker(t)
    try:
        expiries = tk.options
    except Exception:
        return {}
    if not expiries:
        return {}
    now = pd.Timestamp.now().normalize()
    target = now + pd.Timedelta(days=target_days)
    best_exp = min(expiries, key=lambda e: abs(pd.Timestamp(e) - target))
    try:
        chain = tk.option_chain(best_exp)
    except Exception:
        return {}
    calls, puts = chain.calls, chain.puts
    if calls is None or puts is None or calls.empty or puts.empty:
        return {}
    try:
        spot = float(tk.fast_info.last_price)
    except Exception:
        return {}
    if not np.isfinite(spot) or spot <= 0:
        return {}

    def _sane_iv(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["strike", "impliedVolatility"])
        return df[df["impliedVolatility"] > 0.05]

    calls, puts = _sane_iv(calls), _sane_iv(puts)
    if calls.empty:
        return {}

    # ATM IV: median IV of strikes within 5% of spot
    atm = calls[(calls["strike"] - spot).abs() / spot <= 0.05]
    atm_iv = float(atm["impliedVolatility"].median()) if len(atm) else None

    # skew: IV at 90% strike minus IV at 110% strike (interpolate nearest)
    def _iv_at(df: pd.DataFrame, strike: float):
        if df.empty:
            return np.nan
        idx = (df["strike"] - strike).abs().idxmin()
        return float(df.loc[idx, "impliedVolatility"])

    iv_dn = _iv_at(calls, 0.9 * spot)
    iv_up = _iv_at(calls, 1.1 * spot)
    skew = (iv_dn - iv_up) if np.isfinite(iv_dn) and np.isfinite(iv_up) else None

    pv = float(puts["volume"].fillna(0).sum()) if "volume" in puts else np.nan
    cv = float(calls["volume"].fillna(0).sum()) if "volume" in calls else np.nan
    put_call = (pv / cv) if cv and cv > 0 else None

    out = {}
    if atm_iv is not None and 0.05 < atm_iv < 3.0:
        out["atm_iv"] = round(atm_iv, 4)
    if skew is not None and np.isfinite(skew):
        out["skew"] = round(float(skew), 4)
    if put_call is not None and np.isfinite(put_call):
        out["put_call_vol"] = round(float(put_call), 3)
    out["expiry"] = best_exp
    out["spot"] = round(spot, 2)
    return out


def build(tickers: list[str], max_tickers: int = 60) -> pd.DataFrame:
    rows = []
    for t in tickers[:max_tickers]:
        m = ticker_options_metrics(t)
        if m:
            m["ticker"] = t
            m["date"] = pd.Timestamp.now().normalize().date()
            rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = monitored universe")
    ap.add_argument("--max-tickers", type=int, default=60)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        stocks = DATA_DIR / "monitored_stocks.parquet"
        if stocks.exists():
            tickers = sorted(pd.read_parquet(stocks)["ticker"].astype(str).str.upper().unique())
        else:
            tickers = []
    df = build(tickers, max_tickers=args.max_tickers)
    print(f"=== Options skew / put-call ({len(df)} tickers) ===")
    cols = [c for c in ["ticker", "date", "spot", "atm_iv", "skew", "put_call_vol", "expiry"] if c in df.columns]
    print(df[cols].sort_values("ticker").to_string(index=False))
    if args.save and len(df):
        df.to_parquet(OUT)
        print(f"\nWrote {OUT}")
        # Point-in-time history. Options quotes are inherently un-recoverable
        # after the fact (yfinance serves only the live chain), so without this
        # append the `skew` component can NEVER be backtested -- unlike the
        # price/fundamental inputs, no amount of later work can reconstruct it.
        # Stamped with the quote date when present, else the latest price date.
        from snapshot_history import append_history
        stamp = None
        if "date" in df.columns:
            stamp = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.isna(stamp):
                stamp = None
        append_history(df, "options_skew", as_of=stamp)


if __name__ == "__main__":
    main()

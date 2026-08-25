#!/usr/bin/env python3
"""
iv_skew.py — Populate options-implied downside tail risk (IV skew) for the
fragility screen. Pragmatic approach: use what yfinance reliably gives.

Why: yfinance's `impliedVolatility` is often stale/cached and delta coverage
is thin. But the ATM vs OTM put IV ratio is a usable skew proxy when
option chains have some liquidity. We compute a simple, robust skew:

    skew = IV(OTM put, delta ~0.15-0.25) / IV(ATM put)

Higher = steeper downside pricing = more tail risk.

For names with insufficient liquidity (fewer than 5 liquid puts across
nearest 2 expiries), we leave NaN — the fragility screen handles NaN as
neutral (pctile 0.5).

Reads: daily_prices.parquet (spot), monitored_stocks.parquet (universe)
Writes: options_skew.csv  (ticker, date, skew, iv_atm, iv_otm, n_options)
Usage:  python iv_skew.py [--tickers A,B,C] [--max-tickers N]
"""
from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
STOCKS = DATA_DIR / "monitored_stocks.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
OUT = DATA_DIR / "options_skew.parquet"

# ── Black-Scholes delta (put) ────────────────────────────────────────────
def _delta(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0


def _interp(arr, target_delta: float) -> float | None:
    """Linear-interpolate IV at target abs-put-delta from sorted (delta, iv)."""
    if len(arr) < 2:
        return None
    ds = [x[0] for x in arr]
    if target_delta < ds[0] or target_delta > ds[-1]:
        return None
    return float(np.interp(target_delta, ds, [x[1] for x in arr]))


def latest_spot() -> pd.Series:
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    p = p.sort_values("date").groupby("ticker").tail(1)
    return p.set_index("ticker")["close"]


def get_tickers(explicit: str | None, max_n: int | None) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist()) if not stocks.empty else []
    if max_n:
        tickers = tickers[:max_n]
    return tickers


def _compute_skew_from_chain(ticker: str, spot: float, chain, T: float, r: float, S: float) -> float | None:
    """Compute skew = IV(OTM put ~0.25Δ) / IV(ATM put) from a single chain.
    Returns None if insufficient liquidity.
    """
    rows = []
    for o in chain.puts.itertuples():
        K = float(o.strike)
        iv = float(getattr(o, "impliedVolatility", 0) or 0)
        oi = getattr(o, "openInterest", 0) or 0
        vol = getattr(o, "volume", 0) or 0
        if iv <= 1e-4 or K <= 0:
            continue
        if oi == 0 and vol == 0:
            continue
        d = _delta(spot, K, T, 0.04, iv)
        if d is None:
            continue
        rows.append((abs(d), iv, float(K)))
    if len(rows) < 10:
        return None
    rows.sort()
    d25 = _interp(rows, 0.25)
    d_atm = _interp(rows, 0.50)
    if d25 is None or d_atm is None or d_atm <= 0:
        return None
    # Skew ratio: OTM put IV / ATM put IV
    # >1 means downside tail priced richer than ATM
    return d25 / d_atm


def skew_for_ticker(ticker: str, spot: float) -> dict | None:
    """Compute skew from nearest 2 expiries with decent liquidity."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = list(tk.options)
        if not exps:
            return None
        today = date.today()
        chosen = [e for e in exps if (pd.Timestamp(e).date() - today).days >= 14][:2]
        if not chosen:
            chosen = exps[:1]
        r = 0.04
        S = float(spot)
        skew_vals = []
        atm_vals = []
        d25_vals = []
        expiries = []
        for exp in chosen:
            chain = tk.option_chain(exp)
            T = max((pd.Timestamp(exp).date() - today).days, 7) / 365.0
            s = _compute_skew_from_chain(ticker, S, chain, T, 0.04, S)
            if s is not None:
                # For simplicity, just compute skew and ATM directly here
                rows = []
                for o in tk.option_chain(exp).puts.itertuples():
                    K = float(o.strike)
                    iv = float(getattr(o, "impliedVolatility", 0) or 0)
                    oi = getattr(o, "openInterest", 0) or 0
                    vol = getattr(o, "volume", 0) or 0
                    if iv <= 1e-4 or K <= 0:
                        continue
                    if oi == 0 and vol == 0:
                        continue
                    d = _delta(S, K, T, 0.04, iv)
                    if d is None:
                        continue
                    rows.append((abs(d), iv, float(K)))
                if len(rows) >= 10:
                    rows.sort()
                    d25 = _interp(rows, 0.25)
                    atm = _interp(rows, 0.50)
                    if d25 is not None and atm is not None and atm > 0:
                        skew_vals.append(d25 / atm)
                        atm_vals.append(atm)
                        d25_vals.append(d25)
                        expiries.append(exp)
        if not skew_vals:
            return None
        skew = float(np.median(skew_vals))
        skew_vol_pts = round((skew - 1.0) * 100, 2)
        return {
                    "ticker": ticker.upper(),
                    "date": pd.Timestamp(date.today()),
                    "skew": round(skew, 4),
                    "skew_vol_pts": skew_vol_pts,
                    "iv_atm": round(np.median(atm_vals), 4) if atm_vals else None,
                    "iv_otm": round(np.median(d25_vals), 4) if d25_vals else None,
                    "n_options": len(skew_vals),
                    "expiry": expiries[0] if expiries else None,
                }
    except Exception as e:
        print(f"  !! {ticker}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import math
    import numpy as np
    import math

    tickers = get_tickers(args.tickers, args.max_tickers)
    if len(tickers) > 500:
        # Limit to liquid, listed names: fetching options chains from yfinance is
        # slow (per-ticker HTTP), so the full 16k universe does not finish inside
        # the DAG timeout. The fragility screen only needs the investable names.
        try:
            st = pd.read_parquet(STOCKS, columns=["ticker", "instrument_type", "exchange"])
            st["ticker"] = st["ticker"].astype(str).str.upper()
            liq = {"NMS", "NYQ", "NCM", "NGM", "ASE"}
            keep = set(st.loc[st["instrument_type"].eq("stock")
                              & st["exchange"].astype(str).isin(liq), "ticker"])
            tickers = [t for t in tickers if t in keep]
        except Exception as e:
            print(f"  WARNING liquid gate skipped: {e}")
    spot = latest_spot()
    if not tickers:
        print("no tickers")
        return

    existing = set()
    if args.skip_existing and OUT.exists():
        e = pd.read_parquet(OUT)
        e["ticker"] = e["ticker"].astype(str).str.upper()
        existing = set(e["ticker"])

    rows = []
    for t in tickers:
        if t in existing:
            continue
        if t not in spot.index or not np.isfinite(spot[t]) or spot[t] <= 0:
            continue
        r = skew_for_ticker(t, spot[t])
        if r:
            rows.append(r)
            print(f"  {t}: skew={r['skew_vol_pts']:+.1f}vp (ratio={r['skew']:.3f})")
    if not rows:
        print("no skew computed")
        return

    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        keep = old[~old["ticker"].astype(str).str.upper().isin(new["ticker"].astype(str).str.upper())]
        new = pd.concat([keep, new], ignore_index=True)
    new.to_parquet(OUT)
    print(f"\nWrote {OUT} ({len(new)} tickers, {len(rows)} updated)")


if __name__ == "__main__":
    main()
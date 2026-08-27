#!/usr/bin/env python3
"""
vol_target.py — Volatility targeting for position sizing (generalized to all tickers).

Idea: scale weight so the position's *standalone* vol contribution matches a target:

    w* = clip( σ_target / σ_asset , w_min, w_max )

Optional portfolio-vol targeting (marginal, diagonal approx):

    w* = σ_port_target * (risk_budget) / σ_asset

High-beta growth names default to a tighter cap because they sit in the higher-risk
growth sleeve.

Usage:
  python vol_target.py --ticker NVDA
  python vol_target.py --ticker NVDA --target-vol 0.20 --window 20
  python vol_target.py --ticker NVDA --portfolio-vol 0.12 --risk-budget 0.15
  python vol_target.py --growth-sleeve          # all growth_ai names with tighter caps
  python vol_target.py --ticker NVDA --save
"""

from __future__ import annotations

import argparse
from datetime import date
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
CALENDAR = DATA_DIR / "rebalance_calendar.parquet"
OUT = DATA_DIR / "vol_targets.parquet"
OUT_CSV = DATA_DIR / "vol_targets.parquet"

# Defaults tuned for high-beta growth names
DEFAULT_TARGET_VOL = 0.25      # 25% annualized standalone vol target for the *position*
DEFAULT_W_MAX = 0.05           # hard cap 5% of portfolio (tight for high-beta names)
DEFAULT_W_MAX_GROWTH = 0.08    # other growth_ai names
DEFAULT_W_MIN = 0.0
DEFAULT_WINDOW = 21            # ~1 month trading days


def load_prices(ticker: str) -> pd.Series:
    df = pd.read_parquet(PRICES)
    df["date"] = pd.to_datetime(df["date"])
    s = (
        df[df["ticker"] == ticker]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["close"]
        .astype(float)
        .dropna()
    )
    return s


def realized_vol(close: pd.Series, window: int = DEFAULT_WINDOW) -> float:
    """Annualized realized vol from log returns (latest window)."""
    if len(close) < max(10, window // 2):
        rets = np.log(close / close.shift(1)).dropna()
    else:
        rets = np.log(close / close.shift(1)).dropna().iloc[-window:]
    if len(rets) < 5:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(252))


def rolling_vol_series(close: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    rets = np.log(close / close.shift(1))
    return rets.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def target_weight(
    asset_vol: float,
    target_vol: float = DEFAULT_TARGET_VOL,
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
) -> float:
    """Inverse-vol weight capped to [w_min, w_max]."""
    if not np.isfinite(asset_vol) or asset_vol <= 1e-8:
        return w_min
    w = target_vol / asset_vol
    return float(np.clip(w, w_min, w_max))


def turnover_band() -> float:
    """Return the latest turnover_band from rebalance_calendar.csv, or 1.0 if unavailable."""
    if not CALENDAR.exists():
        return 1.0
    cal = pd.read_parquet(CALENDAR)
    if cal.empty:
        return 1.0
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    today = date.today()
    row = cal[cal["date"] <= today].tail(1)
    if row.empty:
        return 1.0
    return float(row.iloc[-1]["turnover_band"])


def apply_turnover_cap(w: float, w_cur: float, band: float) -> float:
    """Cap weight drift to +/- band * w_cur, then return capped weight."""
    if band >= 1.0:
        return w
    lower = max(w_cur - band * w_cur, 0.0)
    upper = w_cur + band * w_cur
    return float(np.clip(w, lower, upper))


def portfolio_context(ticker: str) -> dict:
    out = {
        "shares": 0.0,
        "last_close": float("nan"),
        "market_value": 0.0,
        "current_weight": 0.0,
        "portfolio_value": 0.0,
    }
    if not HOLDINGS.exists():
        return out
    h = pd.read_parquet(HOLDINGS)
    total = float(h["market_value"].sum()) if "market_value" in h.columns else 0.0
    out["portfolio_value"] = total
    row = h[h["ticker"] == ticker]
    if len(row):
        r = row.iloc[0]
        out["shares"] = float(r.get("shares", 0) or 0)
        out["last_close"] = float(r.get("last_close", np.nan))
        out["market_value"] = float(r.get("market_value", 0) or 0)
        out["current_weight"] = float(r.get("weight", 0) or 0) / 100.0 if float(r.get("weight", 0) or 0) > 1 else float(r.get("weight", 0) or 0)
        # weight column in holdings was percent-like (~12.74)
        if out["current_weight"] > 1.0:
            out["current_weight"] = out["current_weight"] / 100.0
    return out


def size_position(
    ticker: str,
    target_vol: float = DEFAULT_TARGET_VOL,
    window: int = DEFAULT_WINDOW,
    w_max: float | None = None,
    w_min: float = DEFAULT_W_MIN,
    portfolio_vol: float | None = None,
    risk_budget: float = 0.15,
) -> dict:
    close = load_prices(ticker)
    if close.empty:
        return {"ticker": ticker, "error": "no prices"}

    sigma = realized_vol(close, window=window)
    vol_path = rolling_vol_series(close, window=window).dropna()
    sigma_median = float(vol_path.median()) if len(vol_path) else sigma
    sigma_p75 = float(vol_path.quantile(0.75)) if len(vol_path) else sigma

    if w_max is None:
        w_max = DEFAULT_W_MAX_GROWTH

    # Primary: standalone vol targeting
    if portfolio_vol is not None:
        # spend `risk_budget` fraction of portfolio vol budget on this name (diag approx)
        # w * sigma ≈ portfolio_vol * risk_budget
        tgt = portfolio_vol * risk_budget
        w_star = target_weight(sigma, target_vol=tgt, w_min=w_min, w_max=w_max)
        method = "portfolio_vol_budget"
        effective_target = tgt
    else:
        w_star = target_weight(sigma, target_vol=target_vol, w_min=w_min, w_max=w_max)
        method = "standalone_vol_target"
        effective_target = target_vol

    # Apply turnover cap from calendar (reduces weight drift in stress regime)
    band = turnover_band()
    w_star = apply_turnover_cap(w_star, 0.0, band)  # current weight handled below

    ctx = portfolio_context(ticker)
    px = ctx["last_close"] if np.isfinite(ctx["last_close"]) else float(close.iloc[-1])
    pv = ctx["portfolio_value"] or 0.0
    target_value = w_star * pv if pv > 0 else float("nan")
    target_shares = target_value / px if pv > 0 and px > 0 else float("nan")
    delta_shares = target_shares - ctx["shares"] if np.isfinite(target_shares) else float("nan")
    delta_value = target_value - ctx["market_value"] if np.isfinite(target_value) else float("nan")

    # Implied position vol at current vs target weight
    pos_vol_current = ctx["current_weight"] * sigma if np.isfinite(sigma) else float("nan")
    pos_vol_target = w_star * sigma if np.isfinite(sigma) else float("nan")

    return {
        "ticker": ticker.upper(),
        "as_of": close.index[-1].date(),
        "last_close": round(px, 4),
        "window": window,
        "realized_vol_ann": round(sigma, 4),
        "realized_vol_median": round(sigma_median, 4),
        "realized_vol_p75": round(sigma_p75, 4),
        "target_vol": round(effective_target, 4),
        "method": method,
        "w_max": w_max,
        "w_min": w_min,
        "weight_target": round(w_star, 4),
        "weight_current": round(ctx["current_weight"], 4),
        "weight_delta": round(w_star - ctx["current_weight"], 4),
        "shares_current": round(ctx["shares"], 6),
        "shares_target": round(target_shares, 6) if np.isfinite(target_shares) else None,
        "shares_delta": round(delta_shares, 6) if np.isfinite(delta_shares) else None,
        "value_current": round(ctx["market_value"], 2),
        "value_target": round(target_value, 2) if np.isfinite(target_value) else None,
        "value_delta": round(delta_value, 2) if np.isfinite(delta_value) else None,
        "portfolio_value": round(pv, 2),
        "position_vol_current": round(pos_vol_current, 4) if np.isfinite(pos_vol_current) else None,
        "position_vol_target": round(pos_vol_target, 4) if np.isfinite(pos_vol_target) else None,
        "capped": bool(abs((target_vol if portfolio_vol is None else effective_target) / sigma - w_star) > 1e-6)
        if sigma and sigma > 0
        else False,
    }


def apply_turnover_cap(w: float, w_cur: float, band: float) -> float:
    """Cap weight drift to +/- band * w_cur, then return capped weight."""
    if band >= 1.0:
        return w
    lower = max(w_cur - band * w_cur, 0.0)
    upper = w_cur + band * w_cur
    return float(np.clip(w, lower, upper))


def growth_ai_tickers() -> list[str]:
    if not STOCKS.exists():
        return ["NVDA"]
    s = pd.read_parquet(STOCKS)
    if "growth_sleeve" in s.columns:
        return s.loc[s["growth_sleeve"] == "growth_ai", "ticker"].tolist()
    if "growth_tech_index" in s.columns:
        return s.loc[s["growth_tech_index"] == True, "ticker"].tolist()[:5]
    return ["NVDA"]


def main():
    ap = argparse.ArgumentParser(description="Volatility targeting (growth names)")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--target-vol", type=float, default=DEFAULT_TARGET_VOL,
                    help="Target annualized vol for the position (standalone mode)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--w-max", type=float, default=None, help="Max portfolio weight")
    ap.add_argument("--w-min", type=float, default=DEFAULT_W_MIN)
    ap.add_argument("--portfolio-vol", type=float, default=None,
                    help="If set, allocate risk_budget share of this portfolio vol to the name")
    ap.add_argument("--risk-budget", type=float, default=0.15,
                    help="Fraction of portfolio vol budget for this name (with --portfolio-vol)")
    ap.add_argument("--growth-sleeve", action="store_true",
                    help="Run for all growth_ai sleeve names")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    tickers = growth_ai_tickers() if args.growth_sleeve else [args.ticker.upper()]
    rows = []
    for t in tickers:
        w_max = args.w_max
        if w_max is None:
            w_max = DEFAULT_W_MAX_GROWTH
        r = size_position(
            t,
            target_vol=args.target_vol,
            window=args.window,
            w_max=w_max,
            w_min=args.w_min,
            portfolio_vol=args.portfolio_vol,
            risk_budget=args.risk_budget,
        )
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = [
        "ticker", "as_of", "last_close", "realized_vol_ann", "target_vol",
        "weight_current", "weight_target", "weight_delta",
        "shares_current", "shares_target", "shares_delta",
        "position_vol_current", "position_vol_target", "capped", "method",
    ]
    show = [c for c in cols if c in df.columns]
    print(df[show].to_string(index=False))

    for _, r in df.iterrows():
        if "error" in r and pd.notna(r.get("error")):
            continue
        print(
            f"\n{r['ticker']}: σ≈{r['realized_vol_ann']:.1%} → w*={r['weight_target']:.2%}"
            f" (now {r['weight_current']:.2%}, Δw={r['weight_delta']:+.2%})"
            f"{' [HIT CAP]' if r.get('capped') else ''}"
        )
        if r.get("shares_delta") is not None and pd.notna(r["shares_delta"]):
            action = "TRIM" if r["shares_delta"] < 0 else "ADD"
            print(f"  {action} {abs(r['shares_delta']):.4f} shares  "
                  f"(target {r['shares_target']:.4f} @ {r['last_close']})")
            print(f"  position vol: {r['position_vol_current']:.2%} → {r['position_vol_target']:.2%}")

    if args.save:
        if OUT_CSV.exists():
            old = pd.read_parquet(OUT_CSV)
            old = old[~old["ticker"].isin(df["ticker"])]
            out = pd.concat([old, df], ignore_index=True)
        else:
            out = df
        out.to_parquet(OUT_CSV)
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), OUT)
        print(f"\nWrote {OUT_CSV} and {OUT}")


if __name__ == "__main__":
    main()
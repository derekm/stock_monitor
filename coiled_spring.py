#!/usr/bin/env python3
"""coiled_spring.py — detect BB/KC squeeze + shakeout → expansion setups.

Signals
-------
1. squeeze_on: BB inside KC for N consecutive days (default ≥10 of last 20)
2. bb_width_pctile: BB width at low percentile vs lookback (default ≤25th)
3. shakeout: close below BB lower band + volume_z ≥ threshold (default 1.5)
4. post_shakeout_reclaim: close back inside BB within N days (default 5)
5. expansion_confirm: BB width expands ≥X% from squeeze low (default +20%)
6. fundamental_compression: EV/EBITDA compressing, ROIC stable/rising, low debt

Usage:
  python coiled_spring.py --ticker FTNT --asof 2026-04-15
  python coiled_spring.py --universe --asof 2026-04-15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "daily_prices.parquet"
FUND = ROOT / "fundamentals.parquet"


def _load_ticker(ticker: str) -> pd.DataFrame:
    px = pd.read_parquet(DATA, columns=["date", "ticker", "close", "open", "high", "low", "volume"])
    px["date"] = pd.to_datetime(px["date"])
    df = px[px["ticker"] == ticker].sort_values("date").set_index("date")
    return df


def _load_fundamentals(ticker: str) -> pd.DataFrame:
    if not FUND.exists():
        return pd.DataFrame()
    f = pd.read_parquet(FUND, columns=[
        "ticker", "as_of_date", "source", "roe", "roic", "ev_ebitda",
        "debt_to_equity", "interest_coverage", "earnings_stability"
    ])
    f["as_of_date"] = pd.to_datetime(f["as_of_date"])
    return f[f["ticker"] == ticker].sort_values("as_of_date")


def _fundamental_signals(fund: pd.DataFrame, price_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Map quarterly fundamentals to daily price index; forward-fill."""
    if fund.empty:
        return pd.DataFrame(index=price_idx)
    # Keep only real EDGAR/yfinance rows (no seeds)
    seed_src = {"seed_approx_buffett", "seed_aero_dual", "seed_starlink_launch",
                "seed_neardual_spcx", "seed_defensive_etf", "approx_seed_2026-07",
                "stub_growth", "fundamentals_history_backfill"}
    real = fund[~fund["source"].isin(seed_src)].copy()
    if real.empty:
        return pd.DataFrame(index=price_idx)
    real = real.set_index("as_of_date")
    # Daily align: forward fill to price dates
    daily = real.reindex(price_idx, method="ffill")
    return daily


def _indicators(df: pd.DataFrame, fund_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    d["rv20"] = d["ret"].rolling(20).std() * np.sqrt(252)

    # Bollinger Bands (20, 2)
    d["bb_mid"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_mid"] - 2 * d["bb_std"]
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
    d["bb_pos"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # Keltner Channels (20, 1.5 ATR)
    d["tr"] = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["atr20"] = d["tr"].rolling(20).mean()
    d["kc_mid"] = d["close"].rolling(20).mean()
    d["kc_upper"] = d["kc_mid"] + 1.5 * d["atr20"]
    d["kc_lower"] = d["kc_mid"] - 1.5 * d["atr20"]

    # Squeeze: BB inside KC
    d["squeeze_on"] = (d["bb_upper"] < d["kc_upper"]) & (d["bb_lower"] > d["kc_lower"])

    # Volume
    d["vol20"] = d["volume"].rolling(20).mean()
    d["vol_std20"] = d["volume"].rolling(20).std()
    d["vol_z"] = (d["volume"] - d["vol20"]) / d["vol_std20"]

    # BB width percentile vs 252d lookback
    d["bb_width_p252"] = d["bb_width"].rolling(252).apply(
        lambda x: np.searchsorted(np.sort(x.dropna()), x.iloc[-1]) / len(x.dropna()) if x.notna().sum() > 20 else np.nan,
        raw=False
    )

    # Squeeze persistence
    d["squeeze_20d"] = d["squeeze_on"].rolling(20).sum()

    # Fundamental compression signals
    if fund_daily is not None and not fund_daily.empty:
        for col in ["ev_ebitda", "roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability"]:
            if col in fund_daily.columns:
                d[f"fund_{col}"] = fund_daily[col]
        # EV/EBITDA compression: declining (cheaper) over 4 quarters
        if "fund_ev_ebitda" in d.columns:
            d["ev_ebitda_4q_chg"] = d["fund_ev_ebitda"].pct_change(63)  # ~1 quarter
            d["ev_ebitda_1y_chg"] = d["fund_ev_ebitda"].pct_change(252)
        # ROIC stability / improvement
        if "fund_roic" in d.columns:
            d["roic_1y_chg"] = d["fund_roic"].pct_change(252)
        # Low / stable debt
        if "fund_debt_to_equity" in d.columns:
            d["debt_stable"] = d["fund_debt_to_equity"].rolling(252).std() < 0.5

    return d


def detect_spring(ticker: str, asof: str | None = None,
                  squeeze_days: int = 10, width_pctile: float = 0.25,
                  vol_z_thresh: float = 1.5, reclaim_days: int = 5,
                  expand_pct: float = 0.20) -> dict:
    df = _load_ticker(ticker)
    if asof:
        df = df[df.index <= pd.Timestamp(asof)]
    if len(df) < 252:
        return {"ticker": ticker, "error": "insufficient history"}

    fund = _load_fundamentals(ticker)
    fund_daily = _fundamental_signals(fund, df.index)
    d = _indicators(df, fund_daily)

    # Squeeze active?
    last = d.iloc[-1]
    squeeze_active = last["squeeze_20d"] >= squeeze_days
    width_compressed = last["bb_width_p252"] <= width_pctile if pd.notna(last["bb_width_p252"]) else False

    # Look for shakeout in last 20 days
    recent = d.tail(20).copy()
    shakeout = recent[
        (recent["bb_pos"] < 0) & (recent["vol_z"] >= vol_z_thresh)
    ]
    shakeout_day = shakeout.index[-1] if len(shakeout) else None

    # Reclaim within reclaim_days
    reclaimed = False
    expand_confirmed = False
    width_at_shakeout = None
    width_now = None
    if shakeout_day is not None:
        post = d.loc[shakeout_day:].head(reclaim_days + 1)
        reclaimed = (post["bb_pos"] >= 0).any() if len(post) > 1 else False
        # Expansion check
        width_at_shakeout = d.loc[shakeout_day, "bb_width"]
        width_now = d.iloc[-1]["bb_width"]
        if pd.notna(width_at_shakeout) and pd.notna(width_now) and width_at_shakeout > 0:
            expand_confirmed = (width_now / width_at_shakeout - 1) >= expand_pct

    # Fundamental quality at shakeout / now
    fund_quality = {}
    if fund_daily is not None and not fund_daily.empty:
        for col in ["fund_ev_ebitda", "fund_roe", "fund_roic", "fund_debt_to_equity",
                    "fund_interest_coverage", "fund_earnings_stability"]:
            if col in last and pd.notna(last[col]):
                fund_quality[col] = float(last[col])
            elif col in d.columns:
                # last available
                vals = d[col].dropna()
                if len(vals):
                    fund_quality[col] = float(vals.iloc[-1])

    return {
        "ticker": ticker,
        "as_of": str(d.index[-1].date()),
        "close": float(last["close"]),
        "squeeze_active": bool(squeeze_active),
        "squeeze_20d": int(last["squeeze_20d"]),
        "bb_width": float(last["bb_width"]),
        "bb_width_pctile": float(last["bb_width_p252"]) if pd.notna(last["bb_width_p252"]) else None,
        "width_compressed": bool(width_compressed),
        "shakeout_day": str(shakeout_day.date()) if shakeout_day is not None else None,
        "shakeout_vol_z": float(shakeout.iloc[-1]["vol_z"]) if len(shakeout) else None,
        "shakeout_bb_pos": float(shakeout.iloc[-1]["bb_pos"]) if len(shakeout) else None,
        "reclaimed": bool(reclaimed),
        "expand_confirmed": bool(expand_confirmed),
        "bb_width_now": float(width_now) if width_now is not None else float(last["bb_width"]),
        "bb_width_at_shakeout": float(width_at_shakeout) if width_at_shakeout is not None else None,
        "sprung": bool(squeeze_active and width_compressed and shakeout_day is not None and reclaimed and expand_confirmed),
        **fund_quality,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    if args.ticker:
        res = detect_spring(args.ticker, args.asof)
        for k, v in res.items():
            print(f"  {k}: {v}")
        return

    if args.universe:
        px = pd.read_parquet(DATA, columns=["ticker"])
        tickers = sorted(px["ticker"].unique())
        results = []
        for t in tickers:
            try:
                r = detect_spring(t, args.asof)
                if not r.get("error"):
                    results.append(r)
            except Exception as e:
                print(f"  {t}: ERROR {e}")
        if results:
            df = pd.DataFrame(results)
            sprung = df[df["sprung"]]
            print(f"\nTotal scanned: {len(df)}")
            print(f"Squeeze active: {df['squeeze_active'].sum()}")
            print(f"Width compressed: {df['width_compressed'].sum()}")
            print(f"Shook: {df['shakeout_day'].notna().sum()}")
            print(f"Reclaimed: {df['reclaimed'].sum()}")
            print(f"Expanded: {df['expand_confirmed'].sum()}")
            print(f"\n*** SPRUNG ({len(sprung)}): ***")
            if len(sprung):
                cols = ["ticker", "as_of", "shakeout_day", "close",
                        "fund_ev_ebitda", "fund_roic", "fund_debt_to_equity"]
                cols = [c for c in cols if c in sprung.columns]
                print(sprung[cols].to_string(index=False))
            df.to_parquet(ROOT / "coiled_spring_screen.parquet", index=False)
            print("\nWrote coiled_spring_screen.parquet")


if __name__ == "__main__":
    main()

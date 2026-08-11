#!/usr/bin/env python3
"""
earnings_catalyst.py — Earnings catalyst filter: pre-earnings momentum,
post-earnings drift stats, and IV-vs-realized flag (realized-vol proxy).

Inputs:
  earnings_calendar.parquet  (update_earnings.py)
  daily_prices.parquet       (adj_close)
  monitored_stocks.parquet   (sector, ticker)

Outputs:
  earnings_catalyst_signals.csv   per-ticker: next earnings date, pre-mom
                                  percentile, drift expectation, iv_rich flag
  earnings_drift_stats.csv        surprise bucket → forward 5/20/63d drift (OOS)

Honest-OOS rule: drift stats are computed ONLY from the trailing window
(--drift-window, default 750 trading days) ending at --cutoff, and the
"live" signals report expected drift from those buckets — never an in-sample
fit of the same rows the signal is scored on.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import (
    DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns, to_date_keys,
)

EARN = DATA_DIR / "earnings_calendar.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_SIG = DATA_DIR / "earnings_catalyst_signals.parquet"
OUT_DRIFT = DATA_DIR / "earnings_drift_stats.parquet"


def _load_earnings() -> pd.DataFrame:
    if not EARN.exists():
        return pd.DataFrame(columns=["ticker", "earnings_date", "surprise_pct"])
    df = pd.read_parquet(EARN)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def drift_by_bucket(
    earn: pd.DataFrame,
    prices: pd.DataFrame,
    cutoff: pd.Timestamp,
    drift_window: int = 750,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Forward returns after earnings surprises, bucketed by surprise sign.

    Only earnings in (cutoff - drift_window, cutoff) are used — strictly
    trailing data. Returns (bucket_stats_df, bucket_map) where bucket_map
    maps bucket name → avg drift at 5/20/63d for signal scoring.
    """
    wide = wide_closes(prices).sort_index()
    # keep only rows up to cutoff
    wide = wide[wide.index <= cutoff]
    if len(wide) < 100:
        return pd.DataFrame(), {}

    earn = earn[(earn["earnings_date"] > cutoff - pd.Timedelta(days=drift_window * 1.6)) & (earn["earnings_date"] <= cutoff)]
    if earn.empty or "surprise_pct" not in earn.columns:
        return pd.DataFrame(), {}

    bucket_map: dict[str, dict] = {}
    rows: list[dict] = []
    for label, lo, hi in [
        ("big_beat", 5.0, np.inf),
        ("beat", 0.0, 5.0),
        ("miss", -np.inf, 0.0),
    ]:
        sub = earn[(earn["surprise_pct"] >= lo) & (earn["surprise_pct"] < hi)]
        fwd = {h: [] for h in (5, 20, 63)}
        for _, r in sub.iterrows():
            t = r["earnings_date"]
            tk = r["ticker"]
            if tk not in wide.columns:
                continue
            s = wide[tk].loc[:t]
            if len(s) < 2:
                continue
            p0 = s.iloc[-1]
            for h in (5, 20, 63):
                idx = wide.index.get_indexer([t], method="nearest")[0]
                if idx + h < len(wide):
                    p1 = wide.iloc[idx + h][tk]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        fwd[h].append(p1 / p0 - 1.0)
        stats = {f"drift_{h}d": (float(np.mean(v)) if v else np.nan) for h, v in fwd.items()}
        stats["n_events"] = int(len(sub))
        stats["bucket"] = label
        rows.append(stats)
        bucket_map[label] = {
            "drift_5d": stats["drift_5d"],
            "drift_20d": stats["drift_20d"],
            "drift_63d": stats["drift_63d"],
            "n": stats["n_events"],
        }
    return pd.DataFrame(rows), bucket_map


def pre_earnings_momentum(prices: pd.DataFrame, tickers: list[str], lookback: int = 21) -> pd.Series:
    """Return each ticker's 21d return as percentile of its own 252d distribution."""
    wide = wide_closes(prices).sort_index()
    out: dict[str, float] = {}
    for tk in tickers:
        if tk not in wide.columns:
            continue
        s = wide[tk].dropna()
        if len(s) < 260:
            continue
        rets = s.pct_change().dropna()
        recent = s.iloc[-1] / s.iloc[-1 - lookback] - 1.0
        # percentile of recent vs distribution of rolling lookback returns
        roll = (s / s.shift(lookback) - 1.0).dropna().tail(252)
        pct = float((roll < recent).mean()) if len(roll) else np.nan
        out[tk] = pct
    return pd.Series(out)


def iv_vs_realized(prices: pd.DataFrame, tickers: list[str], short: int = 21, long: int = 63) -> pd.Series:
    """IV-proxy: ratio of short-horizon realized vol to long-horizon realized vol.

    > 1.2 → 'rich' (short-term vol elevated vs its own base → options rich).
    Uses clipped daily returns from adj_close.
    """
    wide = wide_closes(prices).sort_index()
    out: dict[str, float] = {}
    rets = clip_returns(wide.pct_change(), 0.35)
    for tk in tickers:
        if tk not in rets.columns:
            continue
        r = rets[tk].dropna()
        if len(r) < long + 5:
            continue
        sv = r.tail(short).std() * np.sqrt(252)
        lv = r.tail(long).std() * np.sqrt(252)
        out[tk] = float(sv / lv) if lv > 0 else np.nan
    return pd.Series(out)


def atm_implied_vol(tickers: list[str], max_tickers: int = 100) -> pd.Series:
    """ATM implied vol from the nearest-dated options chain (yfinance).

    Real IV: for each ticker, take the nearest expiry, pick the call strike
    closest to last price, use its impliedVolatility. Returns NaN for tickers
    without options (ETFs usually have chains; funds sometimes don't). This is
    the drop-in real-IV feed that replaces the realized-vol proxy.
    """
    import yfinance as yf

    out: dict[str, float] = {}
    for t in tickers[:max_tickers]:
        try:
            tk = yf.Ticker(t)
            expiries = tk.options
            if not expiries:
                continue
            # nearest expiry has degenerate IVs (1e-5); pick the one ~30d out
            now = pd.Timestamp.now().normalize()
            target = now + pd.Timedelta(days=30)
            best_exp = min(expiries, key=lambda e: abs(pd.Timestamp(e) - target))
            chain = tk.option_chain(best_exp).calls
            if chain.empty or "strike" not in chain.columns:
                continue
            spot = tk.fast_info.last_price if hasattr(tk, "fast_info") else None
            if spot is None or not np.isfinite(spot):
                continue
            chain = chain.dropna(subset=["impliedVolatility", "strike"])
            if chain.empty:
                continue
            # yfinance's IV column has degenerate 1e-5 rows even at the ATM
            # strike; use the median IV of sane strikes (>0.05) as the market
            # vol level — far more robust than a single strike.
            sane = chain[chain["impliedVolatility"] > 0.05]
            if sane.empty:
                continue
            iv = float(sane["impliedVolatility"].median())
            if 0.05 < iv < 3.0:  # sanity band
                out[t] = iv
        except Exception:  # noqa: BLE001 - per-ticker failure tolerated
            continue
    return pd.Series(out)


def build(cutoff: pd.Timestamp | None = None, drift_window: int = 750, lookback: int = 21,
          args_max_tickers: int = 100, args_dry_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    earn = _load_earnings()
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    tickers = sorted(stocks["ticker"].astype(str).unique()) if not stocks.empty else []
    prices = load_adj_prices_pandas(tickers=tickers)

    if cutoff is None:
        cutoff = prices["date"].max()

    # Latest earnings date per ticker (upcoming OR most recent)
    earn = earn.sort_values("earnings_date")
    latest = earn.groupby("ticker").tail(1)
    # LAST REPORTED surprise per ticker (non-null) — drives expected drift.
    # Using the upcoming row (surprise NaN) was why most names had no drift.
    reported = earn[earn["surprise_pct"].notna()]
    last_reported = reported.groupby("ticker").tail(1) if len(reported) else pd.DataFrame()

    drift_df, bucket_map = drift_by_bucket(earn, prices, pd.Timestamp(cutoff), drift_window)
    pre_mom = pre_earnings_momentum(prices, tickers, lookback)
    iv_ratio = iv_vs_realized(prices, tickers)
    atm_iv = atm_implied_vol(tickers, max_tickers=args_max_tickers) if not args_dry_run else pd.Series(dtype=float)

    sig_rows: list[dict] = []
    for tk in tickers:
        row: dict = {
            "ticker": tk,
            "next_earnings_date": None,
            "surprise_pct": None,
            "pre_mom_pctile": None,
            "pre_mom_flag": None,
            "iv_vs_realized": None,
            "atm_iv": None,
            "iv_rich": None,
            "expected_drift_20d": None,
            "catalyst_score": None,
        }
        if tk in latest["ticker"].values:
            lr = latest[latest["ticker"] == tk].iloc[0]
            row["next_earnings_date"] = lr["earnings_date"].date()
        if tk in last_reported["ticker"].values:
            lr = last_reported[last_reported["ticker"] == tk].iloc[0]
            row["surprise_pct"] = lr["surprise_pct"]
            # expected drift from trailing bucket stats (OOS-estimated)
            if bucket_map:
                b = "big_beat" if lr["surprise_pct"] >= 5 else ("beat" if lr["surprise_pct"] >= 0 else "miss")
                bm = bucket_map.get(b)
                if bm and pd.notna(bm.get("drift_20d")):
                    row["expected_drift_20d"] = round(float(bm["drift_20d"]), 4)
        if tk in pre_mom.index and pd.notna(pre_mom[tk]):
            row["pre_mom_pctile"] = round(float(pre_mom[tk]), 3)
            row["pre_mom_flag"] = "hot" if pre_mom[tk] >= 0.67 else ("cold" if pre_mom[tk] <= 0.33 else "neutral")
        if tk in iv_ratio.index and pd.notna(iv_ratio[tk]):
            row["iv_vs_realized"] = round(float(iv_ratio[tk]), 3)
        if tk in atm_iv.index and pd.notna(atm_iv[tk]):
            row["atm_iv"] = round(float(atm_iv[tk]), 4)
        # iv_rich: real ATM IV vs realized vol when available; else the
        # short/long realized-vol ratio proxy (>1.2).
        iv_num = atm_iv.get(tk) if tk in atm_iv.index else np.nan
        rv_num = (iv_ratio.get(tk) if tk in iv_ratio.index else np.nan) or np.nan
        if pd.notna(iv_num) and pd.notna(rv_num):
            row["iv_rich"] = bool(iv_num / rv_num > 1.2)
        elif pd.notna(rv_num):
            row["iv_rich"] = bool(rv_num > 1.2)
        else:
            row["iv_rich"] = None
        sig_rows.append(row)

    sig = pd.DataFrame(sig_rows)
    # catalyst_score: rank-normalized blend — 70% OOS expected drift,
    # 30% pre-earnings momentum. Names with no reported surprise get a
    # neutral drift term (0) so momentum alone can't dominate.
    if len(sig):
        sig["_drift_rank"] = sig["expected_drift_20d"].rank(pct=True, na_option="keep").fillna(0.5)
        sig["_mom_rank"] = sig["pre_mom_pctile"].rank(pct=True, na_option="keep").fillna(0.5)
        sig["_has_drift"] = sig["expected_drift_20d"].notna().astype(float)
        score = (0.7 * sig["_drift_rank"] + 0.3 * sig["_mom_rank"]) * sig["_has_drift"]
        sig["catalyst_score"] = score.round(3)
        sig = sig.drop(columns=["_drift_rank", "_mom_rank", "_has_drift"])
    sig = to_date_keys(sig, ["next_earnings_date"])
    return sig, drift_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD; default = last price date")
    ap.add_argument("--drift-window", type=int, default=750, help="Trailing days for drift stats (default 750)")
    ap.add_argument("--lookback", type=int, default=21, help="Pre-earnings momentum lookback (default 21)")
    ap.add_argument("--max-tickers", type=int, default=100, help="Cap for ATM-IV fetch")
    ap.add_argument("--no-iv-fetch", action="store_true", help="Skip live options chain (proxy only)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    sig, drift = build(cutoff=cutoff, drift_window=args.drift_window, lookback=args.lookback,
                       args_max_tickers=args.max_tickers, args_dry_run=args.no_iv_fetch)
    print("=== earnings_drift_stats ===")
    print(drift.to_string(index=False) if not drift.empty else "(no drift data yet)")
    hot = sig[sig["pre_mom_flag"] == "hot"]
    rich = sig[sig["iv_rich"] == True]  # noqa: E712
    print(f"\n=== signals: {len(sig)} tickers; {len(hot)} hot pre-mom; {len(rich)} iv-rich ===")
    print(sig.sort_values("catalyst_score", ascending=False).head(15).to_string(index=False))
    if args.save:
        sig.to_parquet(OUT_SIG)
        drift.to_parquet(OUT_DRIFT)
        print(f"\nWrote {OUT_SIG}\nWrote {OUT_DRIFT}")


if __name__ == "__main__":
    main()

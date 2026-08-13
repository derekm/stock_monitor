#!/usr/bin/env python3
"""backtest_long_hold_entry.py — long-hold ENTRY research (no exit).

Question: can a one-shot ignition (gap + volume, optionally ride-quality
overlays) improve long-term holds vs buy-and-hold?

Honest design (overrides the 100-ticker scratch that printed +11.8% excess):
  - full monitored universe
  - T+1 fill (signal today, capital tomorrow)
  - entry-only: cash until first arming, then hold forever (no ATR exit)
  - TWO excesses:
      excess_t0   — vs BH from the ticker's first price (late-start artifact)
      excess_fair — vs BH from the SAME entry date (the real long-hold question)
  - equal-weight daily portfolio vs equal-weight universe (calendar aligned)

Variants (structural, not threshold-tweaks of the same idea):
  gap_vol          — gap>0 AND volume_z>0.5  (the scratch winner)
  gap_vol_trend    — gap_vol AND close > SMA200
  gap_vol_regime   — gap_vol AND close crossed above SMA200 in last 20d
  gap_vol_fresh    — gap_vol AND no prior gap_vol in 63d (re-arm ignition)
  vol_persist      — volume_z>0.5 two days in a row AND 5d mom>0
  first_ignition   — FIRST gap_vol in the ticker's recorded life

Live screen: names whose first (or fresh) ignition is in the last `--fresh-days`
sessions, joined to shock_ride quality (posture/stack/gate/rec).

Research-only. Not a daily JOB.

Usage:
  python backtest_long_hold_entry.py
  python backtest_long_hold_entry.py --n 250
  python backtest_long_hold_entry.py --fresh-days 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "backtest_long_hold_entry.parquet"
SCREEN = ROOT / "long_hold_entry_screen.parquet"
MIN_DAYS = 252


def _wide():
    px = pd.read_parquet(
        ROOT / "daily_prices.parquet",
        columns=["date", "ticker", "close", "open", "volume"],
    )
    px["date"] = pd.to_datetime(px["date"])
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    open_ = px.pivot(index="date", columns="ticker", values="open").sort_index()
    vol = px.pivot(index="date", columns="ticker", values="volume").sort_index()
    vol = vol.ffill()
    return close, open_, vol


def _universe(close: pd.DataFrame, n: int | None) -> list[str]:
    ok = close.columns[close.notna().sum() >= MIN_DAYS].tolist()
    ok = sorted(ok)
    if n:
        # random-stable slice is worse than full; take first n only for smoke
        ok = ok[:n]
    return ok


def _signals(close, open_, vol):
    prev = close.shift(1)
    gap = open_ / prev - 1.0
    vmean = vol.rolling(20, min_periods=10).mean()
    vstd = vol.rolling(20, min_periods=10).std()
    vz = (vol - vmean) / vstd.replace(0, np.nan)
    ret = close.pct_change()
    mom5 = ret.rolling(5, min_periods=3).sum()
    sma200 = close.rolling(200, min_periods=120).mean()
    above = close > sma200
    crossed = above & (~above.shift(1).fillna(False))
    recently_crossed = crossed.rolling(20, min_periods=1).max().astype(bool)
    gap_vol = (gap > 0) & (vz > 0.5)
    # fresh: no gap_vol in prior 63 days (exclude today)
    prior = gap_vol.shift(1).fillna(False).rolling(63, min_periods=1).max().astype(bool)
    return {
        "gap_vol": gap_vol.fillna(False),
        "gap_vol_trend": (gap_vol & above).fillna(False),
        "gap_vol_regime": (gap_vol & recently_crossed).fillna(False),
        "gap_vol_fresh": (gap_vol & ~prior).fillna(False),
        "vol_persist": (
            (vz > 0.5) & (vz.shift(1) > 0.5) & (mom5 > 0)
        ).fillna(False),
        "first_ignition": gap_vol.fillna(False),  # first True handled later
    }


def _first_true(mask: pd.DataFrame) -> pd.DataFrame:
    """One-shot: True only on the first True per column, else False."""
    cum = mask.astype("int8").cumsum()
    return mask & (cum == 1)


def _hold_from_first(entry: pd.DataFrame) -> pd.DataFrame:
    """T+1: after first True, stay 1 forever. Position on day t uses signal t-1."""
    first = _first_true(entry)
    armed = first.cumsum() > 0
    return armed.shift(1).fillna(False)


def _ticker_stats(close, pos, name):
    ret = close.pct_change()
    rows = []
    for t in close.columns:
        c = close[t]
        r = ret[t]
        p = pos[t].astype(float)
        valid = c.notna()
        if valid.sum() < MIN_DAYS:
            continue
        r = r.where(valid)
        p = p.where(valid, 0.0)
        # terminal wealth from t0
        bh = (1.0 + r.fillna(0.0)).prod() - 1.0
        ride = (1.0 + (r.fillna(0.0) * p.fillna(0.0))).prod() - 1.0
        # fair: start at first in-market day
        in_idx = p[p > 0].index
        if len(in_idx) == 0:
            excess_fair = np.nan
            bh_fair = np.nan
            ride_fair = np.nan
            entry_lag = np.nan
        else:
            start = in_idx[0]
            sl = slice(start, None)
            bh_fair = (1.0 + r.loc[sl].fillna(0.0)).prod() - 1.0
            ride_fair = (1.0 + (r.loc[sl].fillna(0.0) * p.loc[sl].fillna(0.0))).prod() - 1.0
            excess_fair = ride_fair - bh_fair
            first_valid = c.first_valid_index()
            entry_lag = (start - first_valid).days if first_valid is not None else np.nan
        rows.append({
            "ticker": t,
            "rule": name,
            "bh_t0": bh,
            "ride_t0": ride,
            "excess_t0": ride - bh,
            "bh_fair": bh_fair,
            "ride_fair": ride_fair,
            "excess_fair": excess_fair,
            "in_market": float(p.mean()),
            "ever_entered": bool(p.max() > 0),
            "entry_lag_days": entry_lag,
        })
    return pd.DataFrame(rows)


def _ew_portfolio(close, pos):
    ret = close.pct_change()
    have = close.notna()
    # universe EW among names with a return today
    ur = ret.where(have).mean(axis=1)
    # strategy: names with position and a return
    sr = (ret * pos.astype(float)).where(have & pos)
    n_long = (have & pos).sum(axis=1)
    pr = sr.mean(axis=1)
    pr = pr.where(n_long > 0, 0.0)
    # align common dates with at least 20 names
    ok = have.sum(axis=1) >= 20
    ur = ur[ok]
    pr = pr.reindex(ur.index).fillna(0.0)
    def _ann(s):
        if len(s) < 60:
            return np.nan
        wealth = (1.0 + s.fillna(0.0)).prod()
        yrs = len(s) / 252.0
        return float(wealth ** (1.0 / yrs) - 1.0) if yrs > 0 and wealth > 0 else np.nan
    def _dd(s):
        eq = (1.0 + s.fillna(0.0)).cumprod()
        return float((eq / eq.cummax() - 1.0).min())
    return {
        "ew_cagr": _ann(pr),
        "ew_bh_cagr": _ann(ur),
        "ew_cagr_spread": _ann(pr) - _ann(ur),
        "ew_maxdd": _dd(pr),
        "ew_bh_maxdd": _dd(ur),
        "ew_mean_n_long": float(n_long.reindex(ur.index).mean()),
    }


def _event_study(close: pd.DataFrame, entry: pd.DataFrame,
                 horizons=(21, 63, 126, 252)) -> dict:
    """When the signal fires, next-H-day log return vs equal-weight universe.

    T+1: event day t uses forward window (t+1, t+H].
    """
    ret = close.pct_change()
    logret = np.log1p(ret)
    ew = logret.mean(axis=1)
    fire = entry.fillna(False)
    out = {}
    for h in horizons:
        fwd = logret.shift(-1).rolling(h).sum().shift(-(h - 1))
        ew_fwd = ew.shift(-1).rolling(h).sum().shift(-(h - 1))
        xs = fwd.sub(ew_fwd, axis=0)
        hits = fire & xs.notna()
        vals = xs.where(hits).stack().dropna()
        if len(vals) == 0:
            out[f"evt_{h}d_n"] = 0
            out[f"evt_{h}d_xs"] = np.nan
            out[f"evt_{h}d_hit"] = np.nan
            out[f"evt_{h}d_ann"] = np.nan
            continue
        out[f"evt_{h}d_n"] = int(len(vals))
        out[f"evt_{h}d_xs"] = float(vals.mean())
        out[f"evt_{h}d_hit"] = float((vals > 0).mean())
        out[f"evt_{h}d_ann"] = float(vals.mean() * (252.0 / h))
    return out


def _live_screen(close, signals, fresh_days: int) -> pd.DataFrame:
    last = close.index.max()
    window = close.index[close.index >= last - pd.Timedelta(days=fresh_days + 10)]
    # last `fresh_days` sessions
    sess = close.dropna(how="all").index[-fresh_days:]
    gv = signals["gap_vol"].reindex(sess)
    gf = signals["gap_vol_fresh"].reindex(sess)
    gt = signals["gap_vol_trend"].reindex(sess)
    first = _first_true(signals["gap_vol"])
    first_w = first.reindex(sess)
    rows = []
    for t in close.columns:
        if t not in gv.columns:
            continue
        fired = gv[t].fillna(False)
        if not fired.any():
            continue
        last_fire = fired[fired].index.max()
        c = close[t]
        rows.append({
            "ticker": t,
            "last_ignition": last_fire.date(),
            "days_ago": int((last - last_fire).days),
            "fresh_ignition": bool(gf[t].fillna(False).any()),
            "trend_ignition": bool(gt[t].fillna(False).any()),
            "lifetime_first": bool(first_w[t].fillna(False).any()),
            "close": float(c.dropna().iloc[-1]) if c.notna().any() else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["lifetime_first", "fresh_ignition", "days_ago"],
                                         ascending=[False, False, True])
    ride_path = ROOT / "shock_ride_tickers.parquet"
    if ride_path.exists():
        sr = pd.read_parquet(ride_path)
        keep = [c for c in [
            "ticker", "recommendation", "fractal_posture", "fractal_stack_depth",
            "ride_gate_open", "long_ride_score", "as_of",
        ] if c in sr.columns]
        out = out.merge(sr[keep], on="ticker", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="cap tickers (smoke)")
    ap.add_argument("--fresh-days", type=int, default=10)
    args = ap.parse_args()

    close, open_, vol = _wide()
    tickers = _universe(close, args.n)
    close, open_, vol = close[tickers], open_[tickers], vol[tickers]
    print(f"Universe {len(tickers)} tickers, {len(close)} dates "
          f"{close.index.min().date()} → {close.index.max().date()}")

    sigs = _signals(close, open_, vol)
    # first_ignition is first gap_vol only
    sigs["first_ignition"] = _first_true(sigs["gap_vol"])

    frames = []
    summaries = []
    for name, mask in sigs.items():
        pos = _hold_from_first(mask)
        stats = _ticker_stats(close, pos, name)
        frames.append(stats)
        ew = _ew_portfolio(close, pos)
        evt = _event_study(close, mask)
        fair = stats["excess_fair"].dropna()
        summaries.append({
            "rule": name,
            "n": int(len(stats)),
            "entered": int(stats["ever_entered"].sum()),
            "mean_excess_t0": float(stats["excess_t0"].mean()),
            "mean_excess_fair": float(fair.mean()) if len(fair) else np.nan,
            "hit_fair": float((fair > 0).mean()) if len(fair) else np.nan,
            "median_entry_lag_days": float(stats["entry_lag_days"].median()),
            "mean_in_market": float(stats["in_market"].mean()),
            **ew,
            **evt,
        })
        print(f"  {name}: entered {summaries[-1]['entered']}/{summaries[-1]['n']}  "
              f"EW CAGR {ew['ew_cagr_spread']:+.3f}  "
              f"evt252 ann {evt.get('evt_252d_ann', float('nan')):+.3f}  "
              f"lag {summaries[-1]['median_entry_lag_days']:.0f}d")

    per = pd.concat(frames, ignore_index=True)
    summ = pd.DataFrame(summaries)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== LONG-HOLD ENTRY (full available universe, T+1, no exit) ===")
    print(summ.to_string(index=False))
    per.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT} ({len(per)} rows)")

    screen = _live_screen(close, sigs, args.fresh_days)
    if len(screen):
        screen.to_parquet(SCREEN, index=False)
        print(f"Wrote {SCREEN} ({len(screen)} recent ignitions)")
        show = screen.head(25)
        cols = [c for c in [
            "ticker", "last_ignition", "days_ago", "fresh_ignition",
            "lifetime_first", "trend_ignition", "recommendation",
            "fractal_posture", "fractal_stack_depth", "ride_gate_open",
            "long_ride_score",
        ] if c in show.columns]
        print("\n=== Live long-hold ignition screen (most recent) ===")
        print(show[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

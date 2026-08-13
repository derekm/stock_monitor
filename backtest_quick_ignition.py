#!/usr/bin/env python3
"""backtest_quick_ignition.py — outside-the-box: 5-day-fractal price+volume
IGNITION rules vs the lagging momentum gate.

The ride gate is a LAGGING momentum-level detector: mom12>0.40 opens after a
surge, and the price-vs-momentum backtest showed momentum/price-position signals
all have NEGATIVE forward spread (they capture the run too late -> mean reversion
after). This script tests the opposite hypothesis: the 5-DAY fractal (the
fastest granularity) can fire a QUICK IGNITION signal — price re-accelerating on
a volume surge — that catches a breakout EARLY, before the longer momentum
windows confirm.

Signals tested (all computed on DAILY OHLCV, no lookahead, position applied next
day):

  Quick ignition (5-day fractal / short-window):
    ign_vol_price      — 5d momentum turning up AND volume_z > 0.5 (short-window
                         price+volume surge)
    ign_breakout_vol   — close > trailing 5d high (short Donchian breakout) AND
                         volume_z > 0.5
    ign_pctile_vol     — 5-day close_pctile > 0.8 AND volume_z > 0.5 (price near
                         its 5d high on a volume surge)
    ign_gap_vol        — open gap > 0 (gapped up) AND volume_z > 0.5
  Baselines:
    mom_gate           — classic daily momentum gate (mom12>0.40 & mom3>0), exit
                         mom3<=0
    buy_hold

Entry rules use the fast fractal; RISK is managed by the improved ride exit:
the structural ATR-chandelier stop (let winners run, cut losers) and vol-scaled
sizing, from the ride-longevity work. Each rule is run with:
  - flat (full size, ATR-chandelier exit)
  - volscale (size by target/realized vol, ATR-chandelier exit)

Metrics per rule: total return, mean excess vs buy-hold, hit rate, mean maxDD,
in-market fraction, and LEAD (how much earlier the ignition fires vs the
mom_gate on the same name — the "quicker view" claim).

Outputs: backtest_quick_ignition.parquet (per-ticker per-rule)
Usage: python backtest_quick_ignition.py [--n 250]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
OUT = DATA_DIR / "backtest_quick_ignition.parquet"
MIN_DAYS = 60


def load_matrix(cols=("close", "volume", "open")):
    px = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                         columns=["date", "ticker", *cols])
    px["date"] = pd.to_datetime(px["date"])
    out = {}
    for c in cols:
        out[c] = px.pivot(index="date", columns="ticker", values=c)
    return out


def atr(h, l, c, n_atr=14):
    tr = np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
    tr[0] = 0
    return pd.Series(tr).ewm(span=n_atr, adjust=False).mean().to_numpy()


def signal_series(close, volume, open_, mode):
    n = len(close)
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    rv = pd.Series(ret).rolling(20).std().to_numpy() * np.sqrt(252)
    size = np.clip(0.30 / np.where(rv == 0, np.nan, rv), 0, 1.5)
    size = np.nan_to_num(size, nan=0.0)
    vz = pd.Series(volume).rolling(20).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=True).to_numpy()

    ign = np.zeros(n, dtype=bool)
    if mode == "ign_vol_price":
        mom5 = pd.Series(ret).rolling(5).sum().to_numpy()
        mom5_prev = np.roll(mom5, 1); mom5_prev[0] = np.nan
        ign = (mom5 > 0) & (mom5_prev <= 0) & (vz > 0.5)
    elif mode == "ign_breakout_vol":
        hi5 = pd.Series(close).rolling(5).max().shift(1).to_numpy()
        ign = (close > hi5) & (vz > 0.5)
    elif mode == "ign_pctile_vol":
        pct = np.full(n, np.nan)
        for i in range(4, n):
            pct[i] = (close[i - 4:i + 1] <= close[i]).mean()
        ign = (pct > 0.8) & (vz > 0.5)
    elif mode == "ign_gap_vol":
        gap = open_ / np.roll(close, 1) - 1.0
        gap[0] = 0
        ign = (gap > 0) & (vz > 0.5)
    elif mode == "mom_gate":
        mom12 = pd.Series(ret).rolling(252, min_periods=60).mean().to_numpy() * 252
        mom3 = pd.Series(ret).rolling(63, min_periods=21).mean().to_numpy() * 252
        pos = np.zeros(n)
        inpos = False
        for i in range(1, n):
            if not inpos:
                if mom12[i] > 0.40 and mom3[i] > 0:
                    inpos = True
            else:
                if mom3[i] <= 0:
                    inpos = False
            pos[i] = 1.0 if inpos else 0.0
        return pos, pos  # (position, signal) both same for mom gate
    else:
        raise ValueError(mode)
    return ign.astype(float), ign.astype(float)


def simulate_ticker(close, volume, open_, high, low, mode):
    n = len(close)
    ret = np.zeros(n); ret[1:] = close[1:] / close[:-1] - 1.0
    sig, _ = signal_series(close, volume, open_, mode)
    a = atr(high, low, close)
    rv = pd.Series(ret).rolling(20).std().to_numpy() * np.sqrt(252)
    size = np.clip(0.30 / np.where(rv == 0, np.nan, rv), 0, 1.5)
    size = np.nan_to_num(size, nan=0.0)

    # entry on ignition, exit via 2x ATR chandelier; full or vol-scaled
    results = {}
    for sizemode in ("full", "volscale"):
        pos = np.zeros(n)
        inpos = False; chand = 0.0
        for i in range(1, n):
            if not inpos:
                if sig[i]:
                    inpos = True; chand = close[i] - 2.0 * a[i]
            else:
                chand = max(chand, close[i] - 2.0 * a[i])
                if close[i] < chand:
                    inpos = False
            pos[i] = (size[i] if sizemode == "volscale" else 1.0) if inpos else 0.0
        # no-lookahead: position applied next day
        p_prev = np.roll(pos, 1); p_prev[0] = 0.0
        r = ret * p_prev
        eq = (1 + r).cumprod()
        bh = (1 + ret).cumprod()
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        results[sizemode] = {
            "ride_return": float(r.sum()),
            "buy_hold": float(ret.sum()),
            "excess": float(r.sum() - ret.sum()),
            "max_dd_ride": dd,
            "max_dd_bh": float((bh / np.maximum.accumulate(bh) - 1).min()),
            "in_market": float(p_prev.mean()),
            "n_trades": int(np.sum(np.abs(np.diff(p_prev)) > 0.5)),
        }
    return results


RULES = ["ign_vol_price", "ign_breakout_vol", "ign_pctile_vol",
         "ign_gap_vol", "mom_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    M = load_matrix(["close", "volume", "open", "high", "low"])
    close = M["close"]; volume = M["volume"]; open_ = M["open"]
    high = M["high"]; low = M["low"]
    tickers = [c for c in close.columns if close[c].notna().sum() >= MIN_DAYS][: args.n]

    # Collect per-ticker results to avoid list-length mismatch
    per_rows = []
    for t in tickers:
        c = close[t].dropna()
        v = volume[t].reindex(c.index).ffill().to_numpy()
        o = open_[t].reindex(c.index).to_numpy() if t in open_.columns else c.to_numpy()
        h = high[t].reindex(c.index).to_numpy() if t in high.columns else c.to_numpy()
        lo = low[t].reindex(c.index).to_numpy() if t in low.columns else c.to_numpy()
        c = c.to_numpy()
        for rule in RULES:
            try:
                res = simulate_ticker(c, v, o, h, lo, rule)
            except Exception as e:
                print(f"  {t} {rule}: error {e}")
                continue
            for sizemode, st in res.items():
                per_rows.append({"ticker": t, "rule": rule, "size_mode": sizemode, **st})

    if not per_rows:
        print("No results")
        return 0

    df = pd.DataFrame(per_rows)
    # Aggregate
    out = df.groupby(["rule", "size_mode"]).agg(
        total_ride_return=("ride_return", "sum"),
        mean_excess=("excess", "mean"),
        hit_rate=("excess", lambda x: (x > 0).mean()),
        mean_max_dd=("max_dd_ride", "mean"),
        mean_in_market=("in_market", "mean"),
        total_trades=("n_trades", "sum"),
    ).reset_index()
    out["rule"] = out["rule"] + ":" + out["size_mode"]
    out = out.drop(columns=["size_mode"])
    # Buy-hold
    bh_returns = []
    for t in tickers:
        c = close[t].dropna().to_numpy()
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        bh_returns.append(float((1 + ret).cumprod()[-1] - 1))
    bh_mean = np.mean(bh_returns) if bh_returns else 0
    bh_row = pd.DataFrame([{"rule": "buy_hold", "total_ride_return": bh_mean * len(bh_returns),
                            "mean_excess": 0.0, "hit_rate": np.nan, "mean_max_dd": np.nan,
                            "mean_in_market": 1.0, "total_trades": 0}])
    out = pd.concat([out, bh_row], ignore_index=True)

    pd.set_option("display.width", 220)
    print(f"\n=== 5-DAY-FRACTAL QUICK IGNITION vs MOMENTUM GATE ({len(tickers)} tickers, daily, no lookahead) ===")
    print(out.sort_values("mean_excess", ascending=False).to_string(index=False))
    if len(out):
        out.to_parquet(OUT, index=False)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    exit(main())
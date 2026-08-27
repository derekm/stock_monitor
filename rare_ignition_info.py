#!/usr/bin/env python3
"""rare_ignition_info.py — does a RARE quality-filtered price run carry
information that quality/value screens miss, or is it just exuberance?

Design
------
Quality/value pick the *business* (preferred / buy_candidates / implied-r).
They cannot tell you *when a name that was uninteresting becomes ownable*,
because they update on fundamentals. In an ongoing market the first bar you
can actually buy is a *market* bar.

Raw gap+volume is not that bar (half a million events, coin-flip). Rare
quality-filtered ignition is:

  1. gap up AND 20d volume z > 0.5          — someone paid up on size
  2. no prior gap_vol in 63 sessions        — not the 40th gap this quarter
  3. close > SMA200                         — the run is with the trend
  4. not already extended (close/SMA200-1 < 0.15) — not a blow-off chase

(3)+(4) are the anti-exuberance pair: sponsored trend, not a vertical melt.

We measure T+1 forward log excess vs equal-weight universe at 21/63/126/252d
for: raw gap_vol, fresh-only, rare (fresh+trend+not-extended), and the
exuberance complement (fresh+trend+extended). If rare > exuberance, the
tape is saying something other than "everyone already piled in."

Live: join today's rare names to buy_candidates + implied_r + shock_ride so
we can see (a) quality/value names the tape is confirming, and (b) tape
names quality/value still call SATELLITE/AVOID — the "first bar" set.

Research-only.

Usage: python rare_ignition_info.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rare_ignition_info.parquet"
LIVE = ROOT / "rare_ignition_live.parquet"


def _wide():
    px = pd.read_parquet(
        ROOT / "daily_prices/",
        columns=["date", "ticker", "close", "open", "volume"],
    )
    px["date"] = pd.to_datetime(px["date"])
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    open_ = px.pivot(index="date", columns="ticker", values="open").sort_index()
    vol = px.pivot(index="date", columns="ticker", values="volume").sort_index().ffill()
    return close, open_, vol


def _event_xs(close, fire, horizons=(21, 63, 126, 252)):
    logret = np.log1p(close.pct_change())
    ew = logret.mean(axis=1)
    fire = fire.fillna(False)
    rows = {}
    for h in horizons:
        fwd = logret.shift(-1).rolling(h).sum().shift(-(h - 1))
        ew_fwd = ew.shift(-1).rolling(h).sum().shift(-(h - 1))
        xs = fwd.sub(ew_fwd, axis=0)
        vals = xs.where(fire & xs.notna()).stack().dropna()
        if len(vals) == 0:
            rows[f"n_{h}"] = 0
            rows[f"xs_{h}"] = np.nan
            rows[f"hit_{h}"] = np.nan
            rows[f"ann_{h}"] = np.nan
            continue
        rows[f"n_{h}"] = int(len(vals))
        rows[f"xs_{h}"] = float(vals.mean())
        rows[f"hit_{h}"] = float((vals > 0).mean())
        rows[f"ann_{h}"] = float(vals.mean() * (252.0 / h))
    return rows


def main():
    close, open_, vol = _wide()
    # drop names with almost no history
    keep = close.columns[close.notna().sum() >= 252]
    close, open_, vol = close[keep], open_[keep], vol[keep]
    print(f"{close.shape[1]} tickers, {len(close)} dates")

    prev = close.shift(1)
    gap = open_ / prev - 1.0
    vmean = vol.rolling(20, min_periods=10).mean()
    vstd = vol.rolling(20, min_periods=10).std()
    vz = (vol - vmean) / vstd.replace(0, np.nan)
    gap_vol = ((gap > 0) & (vz > 0.5)).fillna(False)
    prior = gap_vol.shift(1).fillna(False).rolling(63, min_periods=1).max().astype(bool)
    fresh = gap_vol & ~prior
    sma = close.rolling(200, min_periods=120).mean()
    ext = close / sma - 1.0
    trend = (close > sma).fillna(False)
    not_ext = (ext < 0.15).fillna(False)
    extended = (ext >= 0.15).fillna(False)
    rare = fresh & trend & not_ext
    exuberant = fresh & trend & extended
    mom12 = close.pct_change(252)
    # residual-ish: 63d name minus EW
    r63 = np.log1p(close.pct_change()).rolling(63).sum()
    resid63 = r63.sub(r63.mean(axis=1), axis=0)
    under_ew = (resid63 < 0).fillna(False)  # catching up, not leading the tape
    rare_laggard = rare & under_ew

    buckets = {
        "gap_vol_raw": gap_vol,
        "fresh_only": fresh,
        "rare_quality": rare,
        "exuberant_fresh": exuberant,
        "rare_laggard": rare_laggard,
    }
    rows = []
    for name, mask in buckets.items():
        ev = _event_xs(close, mask)
        n_day = int(mask.to_numpy().sum())
        print(f"  {name}: fires={n_day}  "
              f"ann21={ev.get('ann_21', float('nan')):+.3f}  "
              f"ann252={ev.get('ann_252', float('nan')):+.3f}  "
              f"hit252={ev.get('hit_252', float('nan')):.3f}  n252={ev.get('n_252', 0)}")
        rows.append({"bucket": name, "fires": n_day, **ev})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(out.to_string(index=False))

    # live: last 63 sessions (rare buckets fire ~6–50 times a year)
    sess = close.dropna(how="all").index[-63:]
    last = close.index.max()
    live_sets = {
        "exuberant_fresh": exuberant.reindex(sess).fillna(False),
        "fresh_trend": (fresh & trend).reindex(sess).fillna(False),
        "rare_quality": rare.reindex(sess).fillna(False),
    }
    recs = []
    for label, live_mask in live_sets.items():
        for t in close.columns:
            fired = live_mask[t]
            if not fired.any():
                continue
            when = fired[fired].index.max()
            recs.append({
                "bucket": label,
                "ticker": t,
                "ignition": when.date(),
                "days_ago": int((last - when).days),
                "close": float(close[t].dropna().iloc[-1]),
                "ext_sma200": float(ext[t].dropna().iloc[-1]) if ext[t].notna().any() else np.nan,
                "mom12": float(mom12[t].dropna().iloc[-1]) if mom12[t].notna().any() else np.nan,
            })
    live = pd.DataFrame(recs)
    if live.empty:
        print("No live rare ignitions")
        return 0

    sr = pd.read_parquet(ROOT / "shock_ride_tickers.parquet")
    keep_sr = [c for c in [
        "ticker", "name", "sector", "recommendation", "fractal_posture",
        "fractal_stack_depth", "ride_gate_open", "ride_exit_flag",
        "long_ride_score", "fresh_verdict",
    ] if c in sr.columns]
    live = live.merge(sr[keep_sr], on="ticker", how="left")

    pm = pd.read_parquet(ROOT / "preferred_metrics.parquet")
    pm = pm.rename(columns={
        "decision": "qv_decision",
        "composite_score": "qv_composite",
        "quality_score": "qv_quality",
        "value_score": "qv_value",
    })
    keep_pm = [c for c in [
        "ticker", "qv_decision", "qv_composite", "qv_quality", "qv_value",
        "roe", "roic", "ev_ebitda", "pb_ratio", "buffett_pass", "trifecta_pass",
    ] if c in pm.columns]
    live = live.merge(pm[keep_pm], on="ticker", how="left")

    irp = ROOT / "implied_r_screen.parquet"
    if irp.exists():
        ir = pd.read_parquet(irp)
        cols = [c for c in ["ticker", "implied_r_clean_pct", "excess_ret_verdict",
                            "cheap_robust"] if c in ir.columns]
        live = live.merge(ir[cols], on="ticker", how="left")

    live = live.sort_values(["bucket", "days_ago", "long_ride_score"],
                            ascending=[True, True, False])
    live.to_parquet(LIVE, index=False)
    print(f"Wrote {LIVE} ({len(live)} bucket-rows in last 63 sessions)")

    cols = [c for c in [
        "bucket", "ticker", "name", "sector", "ignition", "days_ago", "ext_sma200",
        "recommendation", "fractal_posture", "fractal_stack_depth", "long_ride_score",
        "qv_decision", "qv_composite", "excess_ret_verdict",
    ] if c in live.columns]

    print("\n=== Exuberant-fresh (the bucket with measured edge) ∩ ride BUY/WATCH ===")
    q = live[
        (live["bucket"] == "exuberant_fresh")
        & (live.get("recommendation").isin(["BUY", "WATCH"]))
    ]
    print(q[cols].to_string(index=False) if len(q) else "(none)")

    print("\n=== Exuberant-fresh ∩ preferred INCLUDE_* ===")
    inc = live[
        (live["bucket"] == "exuberant_fresh")
        & (live.get("qv_decision").astype(str).str.startswith("INCLUDE"))
    ]
    print(inc[cols].to_string(index=False) if len(inc) else "(none)")

    print("\n=== Exuberant-fresh where quality/value is SATELLITE/AVOID/WATCH ===")
    miss = live[
        (live["bucket"] == "exuberant_fresh")
        & (live.get("qv_decision").isin(["AVOID", "SATELLITE", "WATCH"]))
    ].sort_values("long_ride_score", ascending=False)
    print(miss[cols].head(20).to_string(index=False) if len(miss) else "(none)")
    print(f"n miss={len(miss)} / exuberant live={int((live.bucket=='exuberant_fresh').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""momentum_research_backtest.py — backtest the research momentum measures.

Tests each measure's predictive power across the full price universe and finds
reliable confidence thresholds. For each measure and each ticker, at each month
it records whether the signal is ON and the forward k-month return. Aggregates
to hit-rate, mean forward return, and (where a long-only position is implied)
Sharpe. Compares signal-on vs signal-off to find which measures/conditions
actually separate winners from losers.

Measures tested:
  TSMOM 3/6/12 (JFE 2012), JT 6-1 (JT 1993), STMOM 1m (RFS 2022),
  GW-52w high (George-Hwang 2004), and the young-gate.

Usage: python momentum_research_backtest.py [--tickers N] [--window 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_research import (
    tsmom_signal, stmom_1m, gw52_high, jt_momentum, young_gate,
    ENTRY_THRESH, MIN_POST_IPO_MONTHS, VOL_CAP,
)

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "momentum_research_backtest.parquet"


def _monthly_log_returns(wide: pd.DataFrame) -> pd.DataFrame:
    r = np.log(wide / wide.shift(1))
    return r.replace([np.inf, -np.inf], np.nan).resample("ME").sum()


def forward_cum(m: pd.Series, horizon: int) -> pd.Series:
    """Forward horizon-month log return ending at t+horizon, indexed at t."""
    cum = m.cumsum()
    fwd = cum.shift(-horizon) - cum
    return fwd


def _fractal_stack_series(lp: pd.DataFrame, i: int, skip: int,
                          ladders: dict[str, list[int]]) -> dict[str, pd.Series]:
    """PIT fractal stack depth at position i of the log-price matrix.

    For each named ladder, take the ascending full-window lengths and count how
    many consecutive windows (shortest first) are in an uptrend — the same
    leading-run semantics as the persisted-profile stack, but computed directly
    from `lp` so it is usable on the FULL tape (persisted fractal profiles only
    start 2020-09) and honors the JT skip (windows END at i-skip).

    uptrend for a window = ret > 0 AND regression slope of log price > 0
    (matches `fractal_windows.span_uptrend`: ret>0 & slope>0). Both are derived
    from the same window, so no lookahead and no persisted dependency.
    """
    out: dict[str, pd.Series] = {}
    T = lp.shape[1]
    idx = lp.columns
    arr = lp.to_numpy(dtype=float)
    for name, lengths in ladders.items():
        depth = np.zeros(T, dtype=int)
        for L in sorted(lengths):
            e = i - skip          # window end (exclusive anchor)
            a = e - L             # window start
            if a < 0:
                break             # not enough history for this or longer windows
            y = arr[a:e, :]       # positions a..e-1 -> length L (LP values)
            u = np.arange(L, dtype=float)
            # regression slope of log price over the window: cov(u,y)/var(u)
            denom = L * float((u * u).sum()) - float(u.sum()) ** 2
            num = L * (u[:, None] * y).sum(axis=0) - float(u.sum()) * y.sum(axis=0)
            slope = num / denom
            last = arr[e - 1, :]
            first = arr[a, :]
            ret = last - first
            up = (ret > 0) & (slope > 0)
            depth = depth * up.astype(int) + up.astype(int)   # leading-run count
        out[name] = pd.Series(depth, index=idx)
    return out


def jt_ls_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jegadeesh/Titman long-short: 12-1 vs 12-2 vs fractal stack.

    Monthly-rebalance, equal-weight, top/bottom quintile, 10 bps per side
    (20 bps/month round trip on the LS book). Signals:
      mom_12_1  = 252d log return skipping the last 21d (t-252 .. t-21)
      mom_12_2  = 252d log return skipping the last 42d (t-252 .. t-42)
      mom_fractal = stack_depth (consecutive confirmed ladder spans 15/30/45/90)
    vs TMI on the same dates. Bar: 12-2 net LS beats 12-1 by +2 pp/yr, or
    fractal beats both by that amount.
    """
    from macro_sector_shock import _load_price_matrix
    w = _load_price_matrix()
    lp = np.log(w.replace(0, np.nan))
    dates = pd.DatetimeIndex(lp.index)
    month_ends = dates[dates.is_month_end]  # PIT: signal at month-end, hold next month

    # signal panel: one row per (date, ticker) with JT signals + 12-month
    # fractal stacks (b3 ladder and b6 fine view), each with JT skip parity:
    # skip 21 mirrors 12-1 (drop last month), skip 42 mirrors 12-2.
    LADDERS = {
        "b3": [63, 126, 189, 252],              # (21,3)(42,3)(63,3)(84,3) full windows
        "b6": [42, 84, 126, 168, 210, 252],     # (42,6) fine view, 2-month bars
    }
    frames = []
    for t in month_ends:
        i = dates.get_loc(t)
        if i < 252 + 42:
            continue
        part = pd.DataFrame({"ticker": lp.columns})
        part["date"] = t.date()
        part["mom_12_1"] = (lp.iloc[i - 21] - lp.iloc[i - 252]).values
        part["mom_12_2"] = (lp.iloc[i - 42] - lp.iloc[i - 252]).values
        st21 = _fractal_stack_series(lp, i, skip=21, ladders=LADDERS)
        st42 = _fractal_stack_series(lp, i, skip=42, ladders=LADDERS)
        part["mom_fractal_12_b3_21"] = st21["b3"].values
        part["mom_fractal_12_b3_42"] = st42["b3"].values
        part["mom_fractal_12_b6_21"] = st21["b6"].values
        part["mom_fractal_12_b6_42"] = st42["b6"].values
        frames.append(part)
    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    # forward monthly return = next month-end / this month-end (log)
    fwd_rows = []
    for j, t in enumerate(month_ends):
        i = dates.get_loc(t)
        if j + 1 < len(month_ends):
            i2 = dates.get_loc(month_ends[j + 1])
            fwd_rows.append(pd.DataFrame({
                "date": t.date(),
                "ticker": lp.columns,
                "fwd_ret": (lp.iloc[i2] - lp.iloc[i]).values,
            }))
    fwd_df = pd.concat(fwd_rows, ignore_index=True)
    panel = panel.merge(fwd_df, on=["ticker", "date"], how="inner").dropna(subset=["fwd_ret"])
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    # TMI benchmark on the same dates
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    tmi["date"] = pd.to_datetime(tmi["date"]).dt.date
    tmi = tmi.set_index("date")["ret_net"]
    panel["tmi"] = panel["date"].map(lambda d: tmi.get(d, np.nan))

    # long-short quintile per date
    SIGNALS = ["mom_12_1", "mom_12_2",
               "mom_fractal_12_b3_21", "mom_fractal_12_b3_42",
               "mom_fractal_12_b6_21", "mom_fractal_12_b6_42"]
    out = []
    for d, g in panel.groupby("date"):
        row = {"date": d}
        for sig in SIGNALS:
            s = g.dropna(subset=[sig]).copy()
            if len(s) < 40:
                row[sig] = np.nan
                continue
            s["q"] = pd.qcut(s[sig].rank(method="first"), 5, labels=False)
            long_r = float(s.loc[s["q"] == 4, "fwd_ret"].mean())
            short_r = float(s.loc[s["q"] == 0, "fwd_ret"].mean())
            gross = long_r - short_r
            row[sig] = gross - 0.002  # 10 bps per side
        row["tmi"] = float(g["tmi"].mean())
        out.append(row)
    ls = pd.DataFrame(out)
    if ls.empty or "date" not in ls.columns:
        return pd.DataFrame(), pd.DataFrame()
    ls = ls.set_index("date")
    # Bar is defined on the OVERLAPPING tape: cut all signals to dates where
    # every one is present, so 12-2 vs 12-1 vs fractals share one tape.
    ls_overlap = ls.dropna(subset=SIGNALS)
    # annualize (12 rebalances/yr)
    ann = {}
    for sig in SIGNALS:
        s = ls[sig].dropna()
        ann[sig] = {"net_ann": float(s.mean() * 12),
                    "n_months": int(len(s)),
                    "gross_ann": float((s + 0.002).mean() * 12),
                    "net_ann_overlap": float(ls_overlap[sig].mean() * 12) if len(ls_overlap) else np.nan,
                    "n_overlap": int(len(ls_overlap))}
    ann_df = pd.DataFrame(ann).T
    ann_df.index.name = "signal"
    ann_df = ann_df.reset_index()
    ann_df["vs_tmi_net"] = ann_df["net_ann"] - float(ls["tmi"].mean() * 12)
    return ls, ann_df


def run(tickers_cap: int | None = None) -> pd.DataFrame:
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_cap:
        tickers = tickers[:tickers_cap]

    m_all = _monthly_log_returns(w)
    # cross-sectional ADV (liquidity) proxy: mean |monthly return| (turnover proxy)
    adv_proxy = m_all.abs().mean(axis=1)

    rows = []
    for t in tickers:
        m = m_all[t].replace([np.inf, -np.inf], np.nan).dropna()
        if len(m) < 8:
            continue
        # signals (all long-only, proper 0/1 via >0 boolean, not int-truncation)
        ts3 = (tsmom_signal(m, 3, vol_scaled=False) > 0).astype(float)
        ts6 = (tsmom_signal(m, 6, vol_scaled=False) > 0).astype(float)
        ts12 = (tsmom_signal(m, 12, vol_scaled=False) > 0).astype(float)
        jt6 = m.cumsum().diff(6).gt(0).astype(float)          # JT 6-mo formation
        sm1 = (m > 0).astype(float)                           # STMOM 1-mo
        cum = m.cumsum()
        hi12 = cum.rolling(12, min_periods=1).max()
        gw = (cum / hi12).fillna(0) >= 0.90                    # GW near-high
        ann_vol = m.rolling(12).std() * np.sqrt(12)
        vol_ok = ann_vol <= VOL_CAP

        for i in range(len(m) - 1):
            if i < 6:
                continue
            # forward returns at 3/6/12 mo
            for h in (3, 6, 12):
                if i + h >= len(m):
                    continue
                fwd = float((cum.iloc[i + h] - cum.iloc[i]))
                base = {
                    "ticker": t,
                    "date": m.index[i],
                    "horizon": h,
                    "fwd_log_ret": round(fwd, 4),
                }
                # feature signals (current, not lagged — this is an IC-style test)
                base["tsmom_3"] = int(ts3.iloc[i])
                base["tsmom_6"] = int(ts6.iloc[i])
                base["tsmom_12"] = int(ts12.iloc[i])
                base["jt_6"] = int(jt6.iloc[i])
                base["stmom_1"] = int(sm1.iloc[i])
                base["gw_high"] = int(gw.iloc[i])
                base["vol_ok"] = int(vol_ok.iloc[i])
                base["age_mo"] = int(i)
                rows.append(base)

    df = pd.DataFrame(rows)
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hit-rate + mean forward return for each signal, on vs off."""
    feats = ["tsmom_3", "tsmom_6", "tsmom_12", "jt_6", "stmom_1", "gw_high", "vol_ok"]
    out = []
    for f in feats:
        for h in (3, 6, 12):
            sub = df[df["horizon"] == h]
            on = sub[sub[f] == 1]["fwd_log_ret"]
            off = sub[sub[f] == 0]["fwd_log_ret"]
            if len(on) < 50 or len(off) < 50:
                continue
            out.append({
                "feature": f,
                "horizon": h,
                "n_on": len(on),
                "hit_rate_on": round((on > 0).mean(), 3),
                "mean_on": round(on.mean(), 4),
                "mean_off": round(off.mean(), 4),
                "spread": round(on.mean() - off.mean(), 4),
                "annualized_spread": round((on.mean() - off.mean()) * 12 / h, 3),
            })
    r = pd.DataFrame(out)
    r = r.sort_values("annualized_spread", ascending=False)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    ap.add_argument("--window", type=int, default=60, help="min months per ticker")
    ap.add_argument("--jt", action="store_true",
                    help="Phase 2 item 8: JT long-short 12-1 vs 12-2 vs fractal stack")
    args = ap.parse_args()

    if args.jt:
        OUT_JT = DATA_DIR / "momentum_jt.parquet"
        ls, ann = jt_ls_backtest()
        pd.set_option("display.width", 200)
        print("\n=== JT long-short (monthly rebalance, EW quintile, 10 bps/side, vs TMI) ===")
        print(ann.to_string(index=False))
        if ann.empty:
            print("no data")
            return
        FRACS = ["mom_fractal_12_b3_21", "mom_fractal_12_b3_42",
                 "mom_fractal_12_b6_21", "mom_fractal_12_b6_42"]
        d12_2 = float(ann.loc[ann["signal"] == "mom_12_2", "net_ann_overlap"].iloc[0])
        d12_1 = float(ann.loc[ann["signal"] == "mom_12_1", "net_ann_overlap"].iloc[0])
        fracs = {s: float(ann.loc[ann["signal"] == s, "net_ann_overlap"].iloc[0]) for s in FRACS}
        best_frac = max(fracs, key=fracs.get)
        n_ov = int(ann.loc[ann["signal"] == "mom_12_1", "n_overlap"].iloc[0])
        if np.isfinite(d12_2) and np.isfinite(d12_1):
            print(f"\nOVERLAP TAPE ({n_ov} months, all signals present):")
            print(f"  12-1 {d12_1:+.1%} | 12-2 {d12_2:+.1%}")
            for s in FRACS:
                print(f"  {s:24s} {fracs[s]:+.1%}")
            print(f"\nBAR A (12-2 net − 12-1 net ≥ +2 pp/yr): {(d12_2 - d12_1):+.1%} -> "
                  f"{'PASS' if d12_2 - d12_1 >= 0.02 else 'FAIL'}")
            print(f"BAR B (best fractal − best JT ≥ +2 pp/yr): "
                  f"{fracs[best_frac] - max(d12_1, d12_2):+.1%} (best fractal: {best_frac})")
        ann.to_parquet(OUT_JT, index=False)
        print(f"\nWrote {OUT_JT.name}")
        return

    print("Building feature matrix (signal-on vs forward return)...")
    df = run(tickers_cap=args.tickers)
    print(f"  rows: {len(df)} | tickers: {df['ticker'].nunique()}")
    if df.empty:
        print("no data")
        return

    r = report(df)
    pd.set_option("display.width", 200)
    print("\n=== Measure predictive power (signal-on vs signal-off forward log return) ===")
    print(r.to_string(index=False) if not r.empty else "no features met min sample")

    r.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gold_report.py — how a gold miner interacts with the gold price.

Builds a report from three real sources:
  daily_prices/          equity closes
  macro_data/gold_daily.parquet COMEX front-month gold, USD/oz
  macro_data/gold.parquet       monthly gold for the YoY regime split

Sections:
  1. coverage and identity
  2. beta / correlation / R^2 by window, plus rolling stability
  3. asymmetry: does the stock capture gold's upside as well as its downside
  4. regime split by gold YoY, so the relationship is not read off one regime
  5. drawdown behaviour vs gold and vs a peer
  6. cumulative outcome: gold exposure vs the cost of holding the miner

Beta is an OLS slope on overlapping daily log returns: how much the stock moves
per 1% gold move. Descriptive, not predictive.

Usage:
    python gold_report.py --ticker B
    python gold_report.py --ticker B --peer NEM --proxy gold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices/"
GOLD_D = DATA_DIR / "macro_data" / "gold_daily.parquet"
GOLD_M = DATA_DIR / "macro_data" / "gold.parquet"


def load(tickers: list[str]) -> pd.DataFrame:
    """Wide close panel for `tickers` joined to daily gold, inner on date."""
    p = (pl.read_parquet(PRICES, columns=["ticker", "date", "close"])
         .filter(pl.col("ticker").is_in(tickers))
         .filter(pl.col("close") > 0)
         .to_pandas())
    p["date"] = pd.to_datetime(p["date"])
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index()
    g = pd.read_parquet(GOLD_D).rename(columns={"close": "gold"})
    g["date"] = pd.to_datetime(g["date"])
    return w.join(g.set_index("date")["gold"], how="inner")


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """(beta, annualized alpha, correlation)."""
    if len(y) < 20 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a) * 252.0, float(np.corrcoef(x, y)[0, 1])


def max_dd(series: pd.Series) -> float:
    c = series.dropna()
    return float((c / c.cummax() - 1.0).min()) if len(c) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="B")
    ap.add_argument("--peer", default="NEM")
    ap.add_argument("--windows", default="63,252,756,2520")
    args = ap.parse_args()

    t, peer = args.ticker.upper(), args.peer.upper()
    px = load([t, peer])
    px = px.dropna(subset=[t, "gold"])
    r = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan)
    r = r.dropna(subset=[t, "gold"])

    print("=" * 74)
    print(f"{t} vs GOLD (COMEX front month, USD/oz)")
    print("=" * 74)
    print(f"overlap: {len(r):,} trading days  "
          f"{r.index.min().date()} -> {r.index.max().date()}")
    print(f"gold last: ${px['gold'].iloc[-1]:,.2f}/oz    "
          f"{t} last: ${px[t].iloc[-1]:,.2f}")

    # ---- 2. beta by window
    print()
    print("BETA BY WINDOW")
    print(f"  {'window':>8} {'n':>6} {'beta':>7} {'corr':>7} {'R^2':>6} {'annAlpha':>10}")
    for w in [int(v) for v in args.windows.split(",")] + [len(r)]:
        w = min(w, len(r))
        b, a, c = ols(r[t].tail(w).to_numpy(), r["gold"].tail(w).to_numpy())
        lab = "full" if w == len(r) else f"{w}d"
        print(f"  {lab:>8} {w:>6,} {b:>7.3f} {c:>7.3f} {c*c:>6.3f} {a:>9.1%}")

    # rolling stability
    win = 252
    betas, dates = [], []
    yv, xv = r[t].to_numpy(), r["gold"].to_numpy()
    for i in range(win, len(r), 21):
        b, _, _ = ols(yv[i - win:i], xv[i - win:i])
        if np.isfinite(b):
            betas.append(b)
            dates.append(r.index[i - 1])
    bt = np.array(betas)
    print()
    print(f"ROLLING {win}d BETA  ({len(bt)} monthly windows)")
    print(f"  mean {bt.mean():.3f}  std {bt.std():.3f}  "
          f"min {bt.min():.3f}  max {bt.max():.3f}")
    print(f"  windows with beta > 0: {(bt > 0).mean():.1%}   "
          f"beta > 1: {(bt > 1).mean():.1%}")
    print(f"  latest {bt[-1]:.3f} ({dates[-1].date()})")

    # ---- 3. asymmetry
    up = r["gold"] > 0
    dn = r["gold"] < 0
    bu, _, _ = ols(r.loc[up, t].to_numpy(), r.loc[up, "gold"].to_numpy())
    bd, _, _ = ols(r.loc[dn, t].to_numpy(), r.loc[dn, "gold"].to_numpy())
    print()
    print("UPSIDE vs DOWNSIDE CAPTURE")
    print(f"  gold up   days ({int(up.sum()):,}): beta {bu:.3f}")
    print(f"  gold down days ({int(dn.sum()):,}): beta {bd:.3f}")
    print(f"  asymmetry (up - down): {bu - bd:+.3f}"
          f"   {'captures more upside' if bu > bd else 'suffers more downside'}")

    # ---- 4. regime split on gold YoY
    gm = pd.read_parquet(GOLD_M)
    gm["observation_date"] = pd.to_datetime(gm["observation_date"])
    col = [c for c in gm.columns if c != "observation_date"][0]
    gm = gm.set_index("observation_date")[col]
    yoy = (gm / gm.shift(12) - 1).dropna()
    regime = yoy.reindex(r.index, method="ffill")
    bull = regime > 0.10
    flat = (regime >= -0.10) & (regime <= 0.10)
    bear = regime < -0.10
    print()
    print("BY GOLD REGIME (12-month gold change)")
    print(f"  {'regime':<16} {'days':>7} {'beta':>7} {'corr':>7} "
          f"{'stock ann':>10} {'gold ann':>9}")
    for name, mask in [("gold bull >+10%", bull), ("flat -10..+10%", flat),
                       ("gold bear <-10%", bear)]:
        m = mask.fillna(False)
        if m.sum() < 30:
            print(f"  {name:<16} {int(m.sum()):>7,}  (too few days)")
            continue
        b, _, c = ols(r.loc[m, t].to_numpy(), r.loc[m, "gold"].to_numpy())
        sa = r.loc[m, t].mean() * 252
        ga = r.loc[m, "gold"].mean() * 252
        print(f"  {name:<16} {int(m.sum()):>7,} {b:>7.3f} {c:>7.3f} "
              f"{sa:>9.1%} {ga:>8.1%}")

    # ---- 5. drawdowns
    print()
    print("DRAWDOWN (full overlap)")
    print(f"  {t:<6} max drawdown {max_dd(px[t]):>7.1%}")
    print(f"  {'gold':<6} max drawdown {max_dd(px['gold']):>7.1%}")
    if peer in px.columns:
        print(f"  {peer:<6} max drawdown {max_dd(px[peer]):>7.1%}")
    vol_t = r[t].std() * np.sqrt(252)
    vol_g = r["gold"].std() * np.sqrt(252)
    print(f"  annualized vol: {t} {vol_t:.1%}   gold {vol_g:.1%}   "
          f"ratio {vol_t/vol_g:.2f}x")

    # ---- 6. cumulative outcome
    print()
    print("CUMULATIVE (full overlap)")
    for c in [t, "gold"] + ([peer] if peer in px.columns else []):
        s = px[c].dropna()
        tot = s.iloc[-1] / s.iloc[0] - 1
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (1 + tot) ** (1 / yrs) - 1
        print(f"  {c:<6} total {tot:>9.1%}   CAGR {cagr:>6.2%}   over {yrs:.1f}y")

    if peer in r.columns:
        rp = r.dropna(subset=[peer])
        bp, _, cp = ols(rp[peer].to_numpy(), rp["gold"].to_numpy())
        print()
        print(f"PEER {peer}: gold beta {bp:.3f}, corr {cp:.3f}, "
              f"corr to {t} {rp[t].corr(rp[peer]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

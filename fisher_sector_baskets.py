#!/usr/bin/env python3
"""fisher_sector_baskets.py — Fisher-style price indexes for sector baskets inside an index sleeve.

Builds equal-weight sector basket levels from member closes for:
  - sp500 (by sp500_sector)
  - defensive / growth / fertilizer / portfolio (by GICS sector)

Also writes a long panel for the dashboard Fisher tab.

Usage:
  python fisher_sector_baskets.py --index sp500 --save
  python fisher_sector_baskets.py --index all --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import load_prices_pandas, wide_closes
from index_registry import parse_indexes, tickers_for_index

DATA_DIR = Path(__file__).resolve().parent
STOCKS = DATA_DIR / "monitored_stocks.parquet"
SP500 = DATA_DIR / "sp500_sleeve.csv"
OUT = DATA_DIR / "fisher_sector_baskets.csv"
OUT_LATEST = DATA_DIR / "fisher_sector_baskets_latest.csv"


def members_by_sector(index_name: str) -> dict[str, list[str]]:
    if index_name == "sp500" and SP500.exists():
        sp = pd.read_csv(SP500)
        g = sp.groupby(sp["sp500_sector"].fillna("Unknown"))["ticker"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
        return g
    # generic: membership then sector from monitored_stocks
    ticks = set(tickers_for_index(index_name))
    if not ticks and STOCKS.exists():
        st = pd.read_parquet(STOCKS)
        ticks = set(st["ticker"].astype(str))
    st = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    if st.empty:
        return {}
    st = st[st["ticker"].astype(str).isin(ticks)]
    col = "sector"
    return st.groupby(st[col].fillna("Unknown"))["ticker"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()


def basket_levels(wide: pd.DataFrame, tickers: list[str], min_names: int = 2) -> pd.Series:
    cols = [t for t in tickers if t in wide.columns]
    if len(cols) < min_names:
        return pd.Series(dtype=float)
    sub = wide[cols].dropna(how="all")
    # equal-weight total return index rebased to 100
    rets = sub.pct_change()
    ew = rets.mean(axis=1)
    lvl = (1 + ew.fillna(0)).cumprod() * 100.0
    lvl = lvl.dropna()
    return lvl


def build(index_name: str, lookback_days: int = 756) -> pd.DataFrame:
    groups = members_by_sector(index_name)
    all_t = sorted({t for ts in groups.values() for t in ts})
    prices = load_prices_pandas(prefer_clean=True, tickers=all_t)
    wide = wide_closes(prices)
    if lookback_days and len(wide) > lookback_days:
        wide = wide.iloc[-lookback_days:]
    rows = []
    for sector, ticks in groups.items():
        lvl = basket_levels(wide, ticks)
        if lvl.empty:
            continue
        for dt, val in lvl.items():
            rows.append({
                "index_name": index_name,
                "sector": sector,
                "date": pd.Timestamp(dt).date().isoformat(),
                "level": float(val),
                "n_members": len([t for t in ticks if t in wide.columns]),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="sp500")
    ap.add_argument("--lookback", type=int, default=756)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    names = parse_indexes(args.index) if args.index != "sp500" else ["sp500"]
    # ensure sp500 known
    frames = []
    for n in names:
        try:
            frames.append(build(n, args.lookback))
        except Exception as e:
            print("skip", n, e)
    # always try sp500
    if "sp500" not in names:
        try:
            frames.append(build("sp500", args.lookback))
        except Exception as e:
            print("sp500", e)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print("No basket data")
        return
    # latest snapshot
    latest = df.sort_values("date").groupby(["index_name", "sector"], as_index=False).tail(1)
    # 63d change
    piv = df.pivot_table(index="date", columns=["index_name", "sector"], values="level")
    chg = {}
    if len(piv) > 63:
        last = piv.iloc[-1]
        past = piv.iloc[-64]
        for col in piv.columns:
            if pd.notna(last[col]) and pd.notna(past[col]) and past[col]:
                chg[col] = float(last[col] / past[col] - 1)
    latest["ret_63d"] = latest.apply(lambda r: chg.get((r["index_name"], r["sector"]), np.nan), axis=1)
    print(latest.sort_values("ret_63d", ascending=False).to_string(index=False))
    if args.save:
        df.to_csv(OUT, index=False)
        latest.to_csv(OUT_LATEST, index=False)
        print(f"Wrote {OUT.name} ({len(df)}), {OUT_LATEST.name}")


if __name__ == "__main__":
    main()

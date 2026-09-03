"""Faber GTAA (10-month SMA trend, monthly rebalance) + shareholder yield.

Item 10 of docs/RESEARCH_INTEGRATION_PLAN.md.

Why it exists: `sector_prices` levels are junk, so GTAA is built from real
asset-class ETFs already in the `daily_prices/` hive. Per class we compound
equal-weight member ETF returns into a class index, take the Faber 10-month
SMA trend signal (month-end close > 10-month SMA of month-end closes) and
rebalance monthly: risk assets that are in-trend split the portfolio
equally; everything else goes to cash (BIL if in-trend, else 0% cash leg).
Benchmark is TMI (`bogle_tmi.parquet` ret_net).

Shareholder yield: no buyback column exists in fundamentals, so per the
plan this is dividend yield only — trailing 12m dividends / close, from
`dividends_cache.parquet` (built by fetch_dividends.py via yfinance).

Outputs:
  gtaa_sleeve.parquet          date x asset_class x index, sma, trend_flag, weight
  gtaa_backtest.parquet        date x gtaa_ret, tmi_ret (monthly net, no cost)
  shareholder_yield.parquet    ticker x date x div_yield_ttm
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# Maintained asset-class mapping (ETFs must exist in the daily_prices hive).
ASSET_CLASSES = {
    "us_equity": ["SPY", "IWM"],
    "intl_equity": ["EFA", "VWO"],
    "reits": ["VNQ", "RWR"],
    "bonds": ["AGG", "LQD", "TIP", "HYG"],
    "commodities": ["GLD", "SLV", "DBC", "GSG", "DBA"],
}
CASH_TICKER = "BIL"
SMA_MONTHS = 10


def load_etf_closes(tickers: list[str]) -> pd.DataFrame:
    """Close-price wide frame for the given tickers from the hive."""
    frames = []
    for fp in sorted(glob.glob(str(DATA_DIR / "daily_prices" / "year=*" / "month=*" / "*.parquet"))):
        t = pd.read_parquet(fp, columns=["date", "ticker", "close"])
        t = t[t["ticker"].isin(tickers)]
        if len(t):
            frames.append(t)
    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.drop_duplicates(subset=["date", "ticker"], keep="last")
    wide = px.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide


def class_index(wide: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weight compounded class index from member closes."""
    rets = wide[tickers].pct_change()
    # average member daily return, ignoring members without data yet
    avg = rets.mean(axis=1, skipna=True)
    idx = (1 + avg.fillna(0)).cumprod()
    return idx.rename("index")


def build_sleeve(wide: pd.DataFrame) -> pd.DataFrame:
    """Month-end Faber signals + weights per class (long-only, EW in-trend)."""
    rows = []
    me = wide.resample("ME").last()  # month-end prices
    class_idx = {c: class_index(wide, tks).resample("ME").last() for c, tks in ASSET_CLASSES.items()}
    cash_idx = class_index(wide, [CASH_TICKER]).resample("ME").last()
    for date in me.index:
        flags = {}
        for cls, idx in class_idx.items():
            s = idx.loc[:date].dropna()
            if len(s) < SMA_MONTHS:
                continue
            sma = s.rolling(SMA_MONTHS).mean().iloc[-1]
            flags[cls] = bool(s.iloc[-1] > sma)
        in_trend = [c for c, f in flags.items() if f]
        w_cash = (1.0 - len(in_trend) / len(ASSET_CLASSES)) if in_trend else 1.0
        w_risk = 1.0 / len(ASSET_CLASSES) if in_trend else 0.0
        for cls in ASSET_CLASSES:
            rows.append({
                "date": date.date(),
                "asset_class": cls,
                "index": float(class_idx[cls].loc[date]) if date in class_idx[cls] and not np.isnan(class_idx[cls].loc[date]) else np.nan,
                "trend_flag": flags.get(cls, False),
                "weight": w_risk if flags.get(cls, False) else 0.0,
            })
        rows.append({
            "date": date.date(),
            "asset_class": "cash",
            "index": float(cash_idx.loc[date]) if date in cash_idx.index and not np.isnan(cash_idx.loc[date]) else np.nan,
            "trend_flag": True,
            "weight": w_cash,
        })
    return pd.DataFrame(rows)


def backtest(wide: pd.DataFrame, sleeve: pd.DataFrame, start: str) -> pd.DataFrame:
    """Monthly GTAA return: each class earns its member-EW monthly return
    times its weight set at the previous month-end. Cash leg = BIL return."""
    monthly = wide.resample("ME").last().pct_change()
    cls_mret = {}
    for c, tks in ASSET_CLASSES.items():
        cls_mret[c] = monthly[tks].mean(axis=1, skipna=True)
    cls_mret["cash"] = monthly[[CASH_TICKER]].mean(axis=1, skipna=True)
    mret = pd.DataFrame(cls_mret)

    piv = sleeve.pivot(index="date", columns="asset_class", values="weight")
    piv.index = pd.to_datetime(piv.index)
    # weights decided at month t apply to month t+1 returns
    w = piv.shift(1)
    both = mret.index.intersection(w.index)
    w = w.loc[both]
    mret = mret.loc[both]
    gtaa = (w * mret.fillna(0)).sum(axis=1)

    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    tmi["date"] = pd.to_datetime(tmi["date"])
    tmi_m = tmi.set_index("date")["ret_net"].resample("ME").apply(lambda s: (1 + s).prod() - 1)
    tmi_m = tmi_m.reindex(gtaa.index)
    out = pd.DataFrame({"gtaa_ret": gtaa, "tmi_ret": tmi_m}).loc[start:]
    return out


def perf(bt: pd.DataFrame) -> dict:
    """Same-window comparison: restrict BOTH legs to months where TMI exists
    (TMI starts 2016) so CAGR gaps are apples-to-apples, not window artifacts."""
    res = {}
    common = bt.dropna(subset=["tmi_ret"])
    for col in ["gtaa_ret", "tmi_ret"]:
        r = common[col].dropna()
        nav = (1 + r).cumprod()
        cagr = nav.iloc[-1] ** (12 / len(r)) - 1
        dd = (nav / nav.cummax() - 1).min()
        res[col] = {"cagr": cagr, "maxdd": dd, "n_months": len(r)}
    return res


def window_maxdd(bt: pd.DataFrame, col: str, lo: str, hi: str) -> float:
    r = bt[col].loc[lo:hi].dropna()
    nav = (1 + r).cumprod()
    return float((nav / nav.cummax() - 1).min())


def shareholder_yield(latest_only_date=None) -> pd.DataFrame:
    """Trailing-12m dividend / close panel from dividends_cache.parquet.

    Vectorized: per ticker, cumsum sorted dividends and look up the
    [date-365, date] window with searchsorted — no O(n^2) scan."""
    dc = pd.read_parquet(DATA_DIR / "dividends_cache.parquet")
    dc["ex_date"] = pd.to_datetime(dc["ex_date"])
    px = load_etf_closes(sorted(dc["ticker"].unique()))
    frames = []
    for tk, g in dc.groupby("ticker"):
        if tk not in px.columns:
            continue
        g = g.sort_values("ex_date")
        d = np.asarray(g["ex_date"].values, dtype="datetime64[ns]")
        c = np.concatenate([[0.0], np.cumsum(g["amount"].values)])
        dates = px[tk].dropna().index
        hi = np.searchsorted(d, np.asarray(dates.values), side="right")
        lo = np.searchsorted(d, np.asarray(dates.values) - np.timedelta64(365, "D"), side="right")
        ttm = c[hi] - c[lo]
        close = px[tk].loc[dates].values
        out = pd.DataFrame({"date": [x.date() for x in dates], "ticker": tk,
                            "div_yield_ttm": np.where(close > 0, ttm / close, np.nan)})
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--start", default="2008-01-31")
    args = ap.parse_args()

    tickers = sorted({t for tks in ASSET_CLASSES.values() for t in tks} | {CASH_TICKER})
    print(f"Loading {len(tickers)} ETFs from hive...", flush=True)
    wide = load_etf_closes(tickers)
    print(f"price grid: {wide.index.min().date()} -> {wide.index.max().date()}", flush=True)

    sleeve = build_sleeve(wide)
    bt = backtest(wide, sleeve, args.start)
    perf_res = perf(bt)

    print("\n=== Faber GTAA (10m SMA, monthly, EW in-trend classes, cash=BIL) ===")
    for k, v in perf_res.items():
        print(f"{k}: CAGR {v['cagr']:+.2%} | maxDD {v['maxdd']:.2%} | {v['n_months']} months")
    for lo, hi in [("2020-01-31", "2020-12-31"), ("2022-01-31", "2022-12-31")]:
        g = window_maxdd(bt, "gtaa_ret", lo, hi)
        t = window_maxdd(bt, "tmi_ret", lo, hi)
        ratio = g / t if t else float("nan")
        print(f"{lo[:4]} window maxDD: GTAA {g:.2%} vs TMI {t:.2%} -> ratio {ratio:.2f} (bar < 0.70)")
    cagr_gap = perf_res["gtaa_ret"]["cagr"] - perf_res["tmi_ret"]["cagr"]
    print(f"Full-sample CAGR gap vs TMI: {cagr_gap:+.2%} (bar within 2 pp)")

    if args.save:
        sleeve.to_parquet(DATA_DIR / "gtaa_sleeve.parquet", index=False)
        bt.reset_index().to_parquet(DATA_DIR / "gtaa_backtest.parquet", index=False)
        print("Wrote gtaa_sleeve.parquet, gtaa_backtest.parquet")

    # Shareholder yield (dividend-only leg: no buyback column exists).
    try:
        sy = shareholder_yield()
        last = sy["date"].max()
        cov = sy[sy["date"] == last]
        print(f"\n=== Shareholder yield (div yield TTM only; no buyback data) ===")
        print(f"latest date {last}: {len(cov)} names with px | median yield {cov['div_yield_ttm'].median():.2%}")
        print(f"names with div>0 on latest date: {(cov['div_yield_ttm'] > 0).sum()} (bar >= 500)")
        if args.save:
            sy.to_parquet(DATA_DIR / "shareholder_yield.parquet", index=False)
            print("Wrote shareholder_yield.parquet")
    except FileNotFoundError:
        print("\nNo dividends_cache.parquet — run fetch_dividends.py first")


if __name__ == "__main__":
    main()

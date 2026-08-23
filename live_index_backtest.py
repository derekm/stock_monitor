#!/usr/bin/env python3
"""Parameterized index / sleeve backtest with Sharpe comparison.

Examples:
  python live_index_backtest.py --years 1 --rf 0.04
  python live_index_backtest.py --indexes fertilizer,defensive,portfolio,growth --years 2 --benchmark SPY
  python live_index_backtest.py --years 1 --json   # machine-readable summary for pipeline/dashboard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analytics_common import clip_returns

DATA_DIR = Path(__file__).resolve().parent
OUT_STATS = DATA_DIR / "index_backtest_stats.parquet"
OUT_LEVELS = DATA_DIR / "index_levels_1y.parquet"
OUT_SHARPE = DATA_DIR / "sharpe_comparison.parquet"


def load_prices() -> pd.DataFrame:
    p = DATA_DIR / "daily_prices.parquet"
    df = pd.read_parquet(p)
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    return df


def load_stocks() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "monitored_stocks.parquet")


def load_holdings() -> pd.DataFrame:
    p = DATA_DIR / "portfolio_holdings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=["ticker"])


def member_map(stocks: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, list[str]]:
    def col(name: str) -> list[str]:
        if name not in stocks.columns:
            return []
        return stocks.loc[stocks[name] == True, "ticker"].astype(str).str.upper().tolist()

    port = []
    if "ticker" in holdings.columns:
        port = holdings["ticker"].astype(str).str.upper().tolist()

    return {
        "fertilizer": col("index_member"),
        "defensive": col("defensive_value_index"),
        "growth": col("growth_tech_index"),
        "portfolio": port,
        "personal": port,
    }


def wide_closes(prices: pd.DataFrame, tickers: list[str], start: pd.Timestamp | None) -> pd.DataFrame:
    sub = prices[prices["ticker"].isin(tickers)].copy()
    if start is not None:
        sub = sub[sub["date"] >= start]
    wide = sub.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    return wide


def ew_level(rets: pd.DataFrame, cols: list[str], name: str) -> pd.Series:
    use = [c for c in cols if c in rets.columns]
    if not use:
        return pd.Series(dtype=float, name=name)
    r = rets[use].mean(axis=1, skipna=True)
    lvl = (1 + r.fillna(0)).cumprod() * 100.0
    if len(lvl):
        lvl = lvl / lvl.iloc[0] * 100.0
    lvl.name = name
    return lvl


def perf_stats(series: pd.Series, label: str, n_members: int, rf: float) -> dict | None:
    s = series.dropna()
    if len(s) < 20:
        return None
    r = s.pct_change().dropna()
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    n_years = max((s.index[-1] - s.index[0]).days / 365.25, 0.01)
    ann_ret = (1 + total) ** (1 / n_years) - 1
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    max_dd = float((s / s.cummax() - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")
    return {
        "index": label,
        "start": s.index[0].date().isoformat(),
        "end": s.index[-1].date().isoformat(),
        "total_return_pct": round(total * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(float(sharpe), 3) if sharpe == sharpe else None,
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(float(calmar), 3) if calmar == calmar else None,
        "final_level": round(float(s.iloc[-1]), 2),
        "n_members": n_members,
        "rf": rf,
        "n_days": int(len(s)),
    }


def run(args) -> dict:
    prices = load_prices()
    stocks = load_stocks()
    holdings = load_holdings()
    members = member_map(stocks, holdings)

    want = [x.strip().lower() for x in args.indexes.split(",") if x.strip()]
    label_map = {
        "fertilizer": "Fertilizer",
        "defensive": "Defensive",
        "growth": "GrowthTech",
        "portfolio": "Personal",
        "personal": "Personal",
        "spy": "SPY",
        "benchmark": args.benchmark.upper(),
    }

    end = prices["date"].max()
    start = end - pd.Timedelta(days=int(args.years * 365.25))

    # universe of needed tickers
    tickers: list[str] = []
    for k in want:
        if k in ("spy", "benchmark"):
            tickers.append(args.benchmark.upper())
        else:
            tickers.extend(members.get(k, []))
    tickers = sorted(set(tickers))
    if args.benchmark.upper() not in tickers and args.include_benchmark:
        tickers.append(args.benchmark.upper())

    wide = wide_closes(prices, tickers, start if not args.full_history else None)
    rets = wide.pct_change()
    # Clip extreme daily moves (bad ticks / unadjusted splits) so Sharpe is usable
    clip = float(getattr(args, "clip_daily", 0.35) or 0.35)
    if clip > 0:
        rets = clip_returns(rets, clip)

    levels = {}
    stats = []
    for k in want:
        if k in ("spy", "benchmark"):
            continue
        label = label_map.get(k, k.title())
        cols = members.get(k, [])
        series = ew_level(rets, cols, label)
        if series.empty:
            continue
        levels[label] = series
        st = perf_stats(series, label, len([c for c in cols if c in rets.columns]), args.rf)
        if st:
            stats.append(st)

    if args.include_benchmark:
        b = args.benchmark.upper()
        if b in rets.columns:
            series = ew_level(rets, [b], b)
            levels[b] = series
            st = perf_stats(series, b, 1, args.rf)
            if st:
                stats.append(st)

    if not stats:
        return {"ok": False, "error": "no series built — check indexes / price coverage", "stats": []}

    # Sharpe ranking
    stats_sorted = sorted(stats, key=lambda s: (s.get("sharpe") is not None, s.get("sharpe") or -999), reverse=True)
    for i, s in enumerate(stats_sorted, 1):
        s["sharpe_rank"] = i

    df_stats = pd.DataFrame(stats_sorted)
    df_stats.to_parquet(OUT_STATS)

    # sharpe comparison long form
    sharpe_rows = []
    for s in stats_sorted:
        sharpe_rows.append({
            "index": s["index"],
            "sharpe": s["sharpe"],
            "ann_return_pct": s["ann_return_pct"],
            "ann_vol_pct": s["ann_vol_pct"],
            "max_dd_pct": s["max_dd_pct"],
            "sharpe_rank": s["sharpe_rank"],
            "vs_best_sharpe": round((s["sharpe"] or 0) - (stats_sorted[0]["sharpe"] or 0), 3)
            if stats_sorted[0].get("sharpe") is not None else None,
        })
    pd.DataFrame(sharpe_rows).to_parquet(OUT_SHARPE)

    if levels:
        lvl = pd.concat(levels, axis=1).dropna(how="all")
        lvl = lvl.reset_index().rename(columns={"index": "date"})
        try:
            lvl.to_parquet(OUT_LEVELS, index=False)
        except Exception:
            lvl.to_parquet(DATA_DIR / "index_levels_1y.parquet")

    result = {
        "ok": True,
        "years": args.years,
        "rf": args.rf,
        "benchmark": args.benchmark.upper() if args.include_benchmark else None,
        "start": stats_sorted[0]["start"],
        "end": stats_sorted[0]["end"],
        "stats": stats_sorted,
        "sharpe_comparison": sharpe_rows,
        "files": [str(OUT_STATS.name), str(OUT_SHARPE.name)],
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Parameterized index backtest + Sharpe comparison")
    ap.add_argument("--indexes", default="fertilizer,defensive,portfolio",
                    help="Comma list: fertilizer,defensive,growth,portfolio/personal")
    ap.add_argument("--years", type=float, default=1.0, help="Lookback years (calendar)")
    ap.add_argument("--rf", type=float, default=0.04, help="Annual risk-free rate for Sharpe")
    ap.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    ap.add_argument("--include-benchmark", action="store_true", default=True)
    ap.add_argument("--no-benchmark", action="store_true", help="Skip benchmark series")
    ap.add_argument("--full-history", action="store_true", help="Ignore --years, use all prices")
    ap.add_argument("--clip-daily", type=float, default=0.35,
                    help="Winsorize daily returns to ±this fraction (0=off)")
    ap.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = ap.parse_args()
    if args.no_benchmark:
        args.include_benchmark = False

    result = run(args)
    if args.json:
        print(json.dumps(result, default=str))
    else:
        if not result.get("ok"):
            print("FAIL:", result.get("error"))
            raise SystemExit(1)
        print(f"Backtest {result['start']} → {result['end']}  rf={result['rf']}")
        print(f"Wrote {OUT_STATS.name}, {OUT_SHARPE.name}")
        print("\nSharpe comparison (rank 1 = best):")
        for s in result["stats"]:
            print(
                f"  #{s['sharpe_rank']} {s['index']:12}  Sharpe={s['sharpe']!s:>6}  "
                f"ann={s['ann_return_pct']:+.1f}%  vol={s['ann_vol_pct']:.1f}%  "
                f"maxDD={s['max_dd_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()

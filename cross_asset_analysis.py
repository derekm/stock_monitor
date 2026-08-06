#!/usr/bin/env python3
"""
cross_asset_analysis.py — Cross-asset & cross-sector correlation analysis.

Builds:
  - Sector EW return series (and optional index levels)
  - Full-sample correlation matrices (sector × sector, asset × sector)
  - Rolling correlations for key pairs
  - Stability metrics
  - Asset-to-sector beta / corr tables for portfolio & fertilizer names

Usage:
  python cross_asset_analysis.py all
  python cross_asset_analysis.py sectors
  python cross_asset_analysis.py assets --tickers MOS,CF,SHEL,BAYRY
  python cross_asset_analysis.py rolling --window 20
  python cross_asset_analysis.py save-sector-prices   # writes sector_prices.parquet for forecasting
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
SECTOR_PRICES_FILE = DATA_DIR / "sector_prices.parquet"
SECTOR_CORR_FILE = DATA_DIR / "sector_correlation_matrix.csv"
ASSET_SECTOR_CORR_FILE = DATA_DIR / "asset_sector_correlations.csv"
ROLLING_FILE = DATA_DIR / "rolling_cross_asset_correlations.csv"
STABILITY_FILE = DATA_DIR / "cross_asset_stability.csv"


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PRICES_FILE)
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def load_stocks() -> pd.DataFrame:
    return pd.read_parquet(STOCKS_FILE)


def sector_ew_returns(prices: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    logret = np.log(wide / wide.shift(1))
    out = {}
    for sector, grp in stocks.groupby("sector"):
        cols = [t for t in grp["ticker"] if t in logret.columns]
        if cols:
            out[sector] = logret[cols].mean(axis=1)
    sec = pd.DataFrame(out).dropna(how="all")
    if len(sec) >= 2:
        sec = sec.reindex(pd.bdate_range(sec.index.min(), sec.index.max()))
    return sec


def sector_ew_levels(sec_rets: pd.DataFrame) -> pd.DataFrame:
    """Synthetic EW sector price levels starting at 100."""
    levels = (1 + sec_rets.fillna(0)).cumprod() * 100
    for c in levels.columns:
        first = levels[c].first_valid_index()
        if first is not None:
            levels[c] = levels[c] / levels[c].loc[first] * 100
    return levels


def save_sector_prices(levels: pd.DataFrame) -> None:
    """Long-format sector 'prices' for forecast_granite (ticker = sector name slug)."""
    rows = []
    for sector in levels.columns:
        slug = sector_slug(sector)
        for dt, px in levels[sector].dropna().items():
            rows.append({
                "date": dt.date() if hasattr(dt, "date") else dt,
                "ticker": slug,
                "open": float(px),
                "high": float(px) * 1.002,
                "low": float(px) * 0.998,
                "close": float(px),
                "volume": 0,
                "source": "sector_ew",
                "sector_name": sector,
            })
    df = pd.DataFrame(rows)
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), SECTOR_PRICES_FILE)
    print(f"Wrote {SECTOR_PRICES_FILE} ({len(df)} rows, {df['ticker'].nunique()} sectors)")


def sector_slug(name: str) -> str:
    return "SECT_" + "".join(ch if ch.isalnum() else "_" for ch in name).upper()[:24]


def slug_to_sector(slug: str, stocks: pd.DataFrame) -> str:
    mapping = {sector_slug(s): s for s in stocks["sector"].dropna().unique()}
    return mapping.get(slug, slug.replace("SECT_", "").replace("_", " "))


def cmd_sectors(args):
    prices, stocks = load_prices(), load_stocks()
    sec = sector_ew_returns(prices, stocks)
    corr = sec.corr()
    corr.to_csv(SECTOR_CORR_FILE)
    print("Sector correlation matrix:")
    print(corr.round(2).to_string())
    print(f"\nWrote {SECTOR_CORR_FILE}")

    # Average pairwise corr
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    vals = corr.where(mask).stack()
    print(f"\nMean pairwise sector corr: {vals.mean():.3f}  median={vals.median():.3f}")
    print(f"Most correlated: {vals.idxmax()} = {vals.max():.2f}")
    print(f"Least correlated: {vals.idxmin()} = {vals.min():.2f}")


def cmd_assets(args):
    prices, stocks = load_prices(), load_stocks()
    sec = sector_ew_returns(prices, stocks)
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    logret = np.log(wide / wide.shift(1))

    if getattr(args, "tickers", None):
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = []
        if "in_portfolio" in stocks.columns:
            tickers += stocks[stocks["in_portfolio"] == True]["ticker"].tolist()
        if "index_member" in stocks.columns:
            tickers += stocks[stocks["index_member"] == True]["ticker"].tolist()
        tickers = sorted(set(tickers)) or list(wide.columns[:10])

    rows = []
    print(f"{'Ticker':<8} {'OwnSect':<22} {'corr_own':>8} {'corr_mkt':>8} {'best_other':<22} {'corr':>6}")
    print("-" * 80)
    mkt = sec.mean(axis=1)
    for t in tickers:
        if t not in logret.columns:
            continue
        r = logret[t].dropna()
        own_sector = stocks.loc[stocks["ticker"] == t, "sector"]
        own_sector = own_sector.iloc[0] if len(own_sector) else None
        aligned = pd.concat([r, sec, mkt.rename("mkt")], axis=1).dropna()
        if len(aligned) < 20:
            continue
        corr_own = aligned[t].corr(aligned[own_sector]) if own_sector in aligned.columns else np.nan
        corr_mkt = aligned[t].corr(aligned["mkt"])
        others = {s: aligned[t].corr(aligned[s]) for s in sec.columns if s != own_sector}
        best = max(others, key=others.get) if others else None
        best_c = others.get(best, np.nan) if best else np.nan
        print(f"{t:<8} {str(own_sector)[:22]:<22} {corr_own:8.2f} {corr_mkt:8.2f} "
              f"{str(best)[:22]:<22} {best_c:6.2f}")
        for s in sec.columns:
            rows.append({
                "ticker": t,
                "sector": s,
                "is_home_sector": s == own_sector,
                "corr": round(aligned[t].corr(aligned[s]), 4),
                "n": len(aligned),
            })
        rows.append({
            "ticker": t, "sector": "MARKET_EW", "is_home_sector": False,
            "corr": round(corr_mkt, 4), "n": len(aligned),
        })

    if rows:
        pd.DataFrame(rows).to_csv(ASSET_SECTOR_CORR_FILE, index=False)
        print(f"\nWrote {ASSET_SECTOR_CORR_FILE}")


def cmd_rolling(args):
    prices, stocks = load_prices(), load_stocks()
    sec = sector_ew_returns(prices, stocks)
    w = getattr(args, "window", 20) or 20
    pairs = [
        ("Materials", "Energy"),
        ("Materials", "Consumer Staples"),
        ("Materials", "Health Care"),
        ("Consumer Staples", "Health Care"),
        ("Consumer Staples", "Utilities"),
        ("Energy", "Utilities"),
        ("Financials", "Real Estate"),
        ("Information Technology", "Consumer Discretionary"),
        ("Industrials", "Materials"),
    ]
    out = {}
    for a, b in pairs:
        if a in sec.columns and b in sec.columns:
            label = f"{a[:6]}×{b[:6]}_r{w}"
            out[label] = sec[a].rolling(w, min_periods=max(8, w // 2)).corr(sec[b])
    df = pd.DataFrame(out)
    df.to_csv(ROLLING_FILE)
    print(f"Rolling {w}d cross-sector correlations:")
    print(df.tail(5).round(2).to_string())
    print(f"Wrote {ROLLING_FILE}")

    # stability
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        rows.append({
            "pair": c,
            "mean": round(s.mean(), 4),
            "std": round(s.std(), 4),
            "min": round(s.min(), 4),
            "max": round(s.max(), 4),
            "stability": round(max(0, 1 - s.std()), 4),
        })
    if rows:
        stab = pd.DataFrame(rows).sort_values("std", ascending=False)
        stab.to_csv(STABILITY_FILE, index=False)
        print("\nStability (highest std = least stable):")
        print(stab.to_string(index=False))
        print(f"Wrote {STABILITY_FILE}")


def cmd_save_sector_prices(args):
    prices, stocks = load_prices(), load_stocks()
    sec = sector_ew_returns(prices, stocks)
    levels = sector_ew_levels(sec)
    save_sector_prices(levels)
    # also register sector synthetic tickers into a helper csv
    meta = pd.DataFrame({
        "ticker": [sector_slug(s) for s in levels.columns],
        "sector_name": list(levels.columns),
        "kind": "sector_ew_index",
    })
    meta.to_csv(DATA_DIR / "sector_tickers.csv", index=False)
    print(f"Wrote sector_tickers.csv")


def cmd_all(args):
    cmd_sectors(args)
    print()
    cmd_assets(args)
    print()
    cmd_rolling(args)
    print()
    cmd_save_sector_prices(args)


def main():
    parser = argparse.ArgumentParser(description="Cross-asset / cross-sector analysis")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("all").set_defaults(func=cmd_all)
    sub.add_parser("sectors").set_defaults(func=cmd_sectors)
    p = sub.add_parser("assets")
    p.add_argument("--tickers")
    p.set_defaults(func=cmd_assets)
    p = sub.add_parser("rolling")
    p.add_argument("--window", type=int, default=20)
    p.set_defaults(func=cmd_rolling)
    sub.add_parser("save-sector-prices").set_defaults(func=cmd_save_sector_prices)
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

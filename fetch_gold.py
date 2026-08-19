#!/usr/bin/env python3
"""
fetch_gold.py — gold price series for macro_data/.

Writes two files:

  macro_data/gold.parquet        MONTHLY, schema-compatible with the other
                                 macro_data monthlies: observation_date + a single
                                 value column. Drops straight into
                                 macro_sector_shock.COMMODITY_MAP.
  macro_data/gold_daily.parquet  DAILY, full history, for finer-grained work than
                                 the monthly commodity shock panel allows.

SOURCE

FRED carries no usable USD/oz gold series: the IMF (PGOLDUSDM) and LBMA London
fix (GOLDPMGBD228NLBM, GOLDAMGBD228NLBM) IDs all 404 on the fredgraph CSV
endpoint. What FRED still has is BLS index families -- IQ12260 (Export Price
Index, Nonmonetary Gold) and IR14270 (Import Price Index) -- which are indexes
(Dec 2024=100), not prices, so a "gold beta" against them is not in dollars.

So price comes from the futures/ETF path, and the BLS index is fetched alongside
it as an independent cross-check rather than as the price itself:

  GC=F   COMEX gold futures continuous front month, daily, 2000-08-30 ->
  ^XAU   PHLX Gold/Silver miners INDEX, daily, 1983-12-19 -> (equities, NOT bullion)
  GLD    SPDR Gold Shares ETF, daily, 2004-11-18 ->

GC=F is the default: it is bullion in USD/oz, and it is the longest bullion
history available here. ^XAU reaches back to 1983 but is a MINER index, so it
cannot extend a bullion series -- mixing the two would silently splice equity
returns onto a commodity.

Monthly aggregation uses the LAST observation of each calendar month, stamped to
the first of the month, matching the FRED monthly convention already in
macro_data/ (observation_date = 1992-01-01 for January 1992).

Usage:
    python fetch_gold.py                 # fetch + write both files
    python fetch_gold.py --dry-run       # report coverage, write nothing
    python fetch_gold.py --symbol GC=F
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
MACRO = DATA_DIR / "macro_data"

# value column name for the monthly file; kept short and explicit so downstream
# code reads gold["gold_usd_oz"] rather than a FRED-style opaque ID
MONTHLY_COL = "gold_usd_oz"
BLS_EXPORT_INDEX = "IQ12260"   # BLS export price index, nonmonetary gold
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_daily(symbol: str) -> pd.DataFrame:
    """Daily close history for `symbol` as (date, close)."""
    import yfinance as yf

    h = yf.Ticker(symbol).history(period="max", auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"{symbol}: empty history")
    out = (h[["Close"]].reset_index()
           .rename(columns={"Date": "date", "Close": "close"}))
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0].reset_index(drop=True)
    return out[["date", "close"]]


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end last close, stamped to the FIRST of the month (FRED convention).

    observation_date is written as a YYYY-MM-DD STRING to match the other
    macro_data monthlies on disk; every reader coerces it with pd.to_datetime.
    """
    d = daily.set_index("date").sort_index()
    m = d["close"].resample("MS").last().dropna()
    return pd.DataFrame({
        "observation_date": m.index.strftime("%Y-%m-%d"),
        MONTHLY_COL: m.to_numpy(dtype=float),
    })


def fetch_bls_index() -> pd.DataFrame | None:
    """BLS export price index for nonmonetary gold -- an independent check on the
    futures series, not a price."""
    try:
        d = pd.read_csv(FRED_BASE.format(series=BLS_EXPORT_INDEX))
        d["observation_date"] = pd.to_datetime(d["observation_date"], errors="coerce")
        d = d.dropna(subset=["observation_date"])
        d[BLS_EXPORT_INDEX] = pd.to_numeric(d[BLS_EXPORT_INDEX], errors="coerce")
        return d.dropna()
    except Exception as e:                                     # noqa: BLE001
        print(f"  BLS index unavailable ({type(e).__name__}: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GC=F",
                    help="bullion price symbol (default GC=F, COMEX front month)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"fetching {args.symbol} ...")
    daily = fetch_daily(args.symbol)
    monthly = to_monthly(daily)

    print(f"  daily  : {len(daily):,} rows  "
          f"{daily['date'].min().date()} -> {daily['date'].max().date()}  "
          f"last ${daily['close'].iloc[-1]:,.2f}")
    print(f"  monthly: {len(monthly):,} rows  "
          f"{monthly['observation_date'].min()} -> "
          f"{monthly['observation_date'].max()}")

    # Cross-check against the BLS index the price should co-move with. Compare
    # LEVELS, not month-over-month returns: the monthly stamp carries the month's
    # LAST close while BLS surveys within the month, so a return-on-return
    # comparison is misaligned by up to a month and reads as noise even when the
    # two series are effectively identical.
    bls = fetch_bls_index()
    if bls is not None and len(bls) > 24:
        # monthly.observation_date is a string on disk; align types for the merge
        mm = monthly.assign(
            observation_date=pd.to_datetime(monthly["observation_date"]))
        j = mm.merge(bls, on="observation_date", how="inner")
        if len(j) > 24:
            corr = float(j[MONTHLY_COL].corr(j[BLS_EXPORT_INDEX]))
            print(f"  cross-check vs {BLS_EXPORT_INDEX} (BLS export index): "
                  f"{len(j)} shared months, level corr {corr:.3f}")
            if corr < 0.9:
                print("  WARNING: this should track the BLS gold index almost "
                      "exactly -- verify the symbol is bullion, not miners")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return 0

    MACRO.mkdir(parents=True, exist_ok=True)
    mpath = MACRO / "gold.parquet"
    dpath = MACRO / "gold_daily.parquet"
    monthly.to_parquet(mpath, index=False)
    daily.to_parquet(dpath, index=False)
    print()
    print(f"wrote {mpath.relative_to(DATA_DIR)}  ({len(monthly):,} rows)")
    print(f"wrote {dpath.relative_to(DATA_DIR)}  ({len(daily):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

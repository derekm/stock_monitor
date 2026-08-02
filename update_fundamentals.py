#!/usr/bin/env python3
"""
update_fundamentals.py - Maintain P/B, market cap, total assets, and mktcap/assets ratios.

Usage:
  python update_fundamentals.py show
  python update_fundamentals.py show --ticker CF
  python update_fundamentals.py manual --ticker CF --market-cap-b 18.2 --total-assets-b 13.8 --pb 2.9
  python update_fundamentals.py from-csv fundamentals.csv
  python update_fundamentals.py fetch          # yfinance attempt (marketCap + bookValue when available)

CSV expected columns (minimum):
  ticker, market_cap_b, total_assets_b, pb_ratio
Optional: as_of_date, notes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND_FILE = DATA_DIR / "fundamentals.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def load() -> pd.DataFrame:
    if FUND_FILE.exists():
        return pd.read_parquet(FUND_FILE)
    return pd.DataFrame(columns=[
        "ticker", "as_of_date", "market_cap", "market_cap_b", "total_assets",
        "total_assets_b", "pb_ratio", "mktcap_to_assets", "source", "notes", "last_updated"
    ])


def save(df: pd.DataFrame) -> None:
    df = df.sort_values("ticker").drop_duplicates(subset=["ticker", "as_of_date"], keep="last")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, FUND_FILE)
    print(f"Saved {len(df)} fundamental rows → {FUND_FILE}")


def cmd_show(args):
    df = load()
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
    cols = ["ticker", "as_of_date", "market_cap_b", "total_assets_b", "pb_ratio", "mktcap_to_assets", "source"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\n{len(df)} rows")


def cmd_manual(args):
    df = load()
    t = args.ticker.upper()
    mcap_b = float(args.market_cap_b)
    assets_b = float(args.total_assets_b)
    pb = float(args.pb) if args.pb is not None else None
    ev = float(args.ev_ebitda) if args.ev_ebitda is not None else None
    as_of = pd.to_datetime(args.as_of) if args.as_of else pd.Timestamp.now().normalize()

    # Remove prior row for same ticker+date if present
    df = df[~((df["ticker"] == t) & (df["as_of_date"] == as_of))]

    row = {
        "ticker": t,
        "as_of_date": as_of,
        "market_cap": int(mcap_b * 1e9),
        "market_cap_b": mcap_b,
        "total_assets": int(assets_b * 1e9),
        "total_assets_b": assets_b,
        "pb_ratio": pb,
        "ev_ebitda": ev,
        "mktcap_to_assets": round(mcap_b / assets_b, 3) if assets_b else None,
        "source": "manual",
        "notes": args.notes or "",
        "last_updated": pd.Timestamp.now(),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    print(f"Updated {t}: MktCap ${mcap_b:.2f}B | Assets ${assets_b:.2f}B | P/B {pb} | Mkt/Assets {row['mktcap_to_assets']}")


def cmd_from_csv(args):
    csv_df = pd.read_csv(args.csv)
    required = {"ticker", "market_cap_b", "total_assets_b"}
    if not required.issubset(csv_df.columns):
        raise SystemExit(f"CSV must contain at least: {required}")
    csv_df["ticker"] = csv_df["ticker"].str.upper()
    if "as_of_date" not in csv_df.columns:
        csv_df["as_of_date"] = pd.Timestamp.now().normalize()
    else:
        csv_df["as_of_date"] = pd.to_datetime(csv_df["as_of_date"])
    if "pb_ratio" not in csv_df.columns:
        csv_df["pb_ratio"] = None
    csv_df["market_cap"] = (csv_df["market_cap_b"] * 1e9).astype("int64")
    csv_df["total_assets"] = (csv_df["total_assets_b"] * 1e9).astype("int64")
    csv_df["mktcap_to_assets"] = (csv_df["market_cap_b"] / csv_df["total_assets_b"]).round(3)
    if "source" not in csv_df.columns:
        csv_df["source"] = "csv"
    if "notes" not in csv_df.columns:
        csv_df["notes"] = ""
    csv_df["last_updated"] = pd.Timestamp.now()

    existing = load()
    # Drop overlapping ticker+date
    keys = set(zip(csv_df["ticker"], csv_df["as_of_date"]))
    existing = existing[~existing.apply(lambda r: (r["ticker"], r["as_of_date"]) in keys, axis=1)]
    combined = pd.concat([existing, csv_df], ignore_index=True)
    save(combined)


def cmd_fetch(args):
    """Best-effort yfinance pull for marketCap and bookValue → approximate P/B."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Use manual or from-csv.")
        return

    if STOCKS_FILE.exists():
        tickers = pd.read_parquet(STOCKS_FILE)["ticker"].tolist()
    else:
        tickers = load()["ticker"].tolist()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mcap = info.get("marketCap")
            book = info.get("bookValue")  # per share
            shares = info.get("sharesOutstanding")
            total_assets = None  # rarely in .info; would need balance sheet
            pb = info.get("priceToBook")
            if mcap is None:
                print(f"  {t}: no marketCap")
                continue
            mcap_b = mcap / 1e9
            # If we have bookValue * shares we can estimate equity, not total assets
            row = {
                "ticker": t,
                "as_of_date": pd.Timestamp.now().normalize(),
                "market_cap": int(mcap),
                "market_cap_b": round(mcap_b, 2),
                "total_assets": None,
                "total_assets_b": None,
                "pb_ratio": round(pb, 2) if pb else None,
                "mktcap_to_assets": None,
                "source": "yfinance",
                "notes": f"bookValue/share={book}" if book else "",
                "last_updated": pd.Timestamp.now(),
            }
            rows.append(row)
            print(f"  {t}: MktCap ${mcap_b:.2f}B  P/B {pb}")
        except Exception as e:
            print(f"  {t}: error {e}")

    if not rows:
        print("No data retrieved.")
        return

    new_df = pd.DataFrame(rows)
    existing = load()
    # Prefer keeping existing total_assets if new fetch lacks them
    for _, r in new_df.iterrows():
        prior = existing[existing["ticker"] == r["ticker"]]
        if not prior.empty and r["total_assets"] is None:
            last = prior.sort_values("as_of_date").iloc[-1]
            new_df.loc[new_df["ticker"] == r["ticker"], "total_assets"] = last["total_assets"]
            new_df.loc[new_df["ticker"] == r["ticker"], "total_assets_b"] = last["total_assets_b"]
            if last["total_assets"] and r["market_cap"]:
                new_df.loc[new_df["ticker"] == r["ticker"], "mktcap_to_assets"] = round(
                    r["market_cap"] / last["total_assets"], 3
                )

    keys = set(zip(new_df["ticker"], new_df["as_of_date"]))
    existing = existing[~existing.apply(lambda r: (r["ticker"], r["as_of_date"]) in keys, axis=1)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    save(combined)


def main():
    parser = argparse.ArgumentParser(description="Update P/B, market cap, total assets")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("show")
    p.add_argument("--ticker")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("manual")
    p.add_argument("--ticker", required=True)
    p.add_argument("--market-cap-b", required=True, type=float)
    p.add_argument("--total-assets-b", required=True, type=float)
    p.add_argument("--pb", type=float, default=None)
    p.add_argument("--ev-ebitda", type=float, default=None)
    p.add_argument("--as-of", default=None)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_manual)

    p = sub.add_parser("from-csv")
    p.add_argument("csv")
    p.set_defaults(func=cmd_from_csv)

    p = sub.add_parser("fetch")
    p.add_argument("--tickers", help="Comma-separated subset")
    p.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

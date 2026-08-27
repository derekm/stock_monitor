#!/usr/bin/env python3
"""
portfolio_report.py - Snapshot of holdings, cost basis, and P&L from trades.parquet + latest prices.

  python portfolio_report.py
  python portfolio_report.py --refresh   # recompute holdings from trades + current prices
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
TRADES_FILE = DATA_DIR / "trades.parquet"
HOLDINGS_FILE = DATA_DIR / "portfolio_holdings.parquet"
PRICES_FILE = DATA_DIR / "daily_prices/"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def rebuild_holdings() -> pd.DataFrame:
    trades = pd.read_parquet(TRADES_FILE)
    trades["filled_datetime"] = pd.to_datetime(trades["filled_datetime"])
    buys = trades[trades["transaction_type"].isin(["Buy", "Dividend Reinvestment"])]
    holdings = buys.groupby("ticker").agg(
        shares=("quantity", "sum"),
        cost_basis=("notional", "sum"),
        first_fill=("filled_datetime", "min"),
        last_fill=("filled_datetime", "max"),
        n_trades=("trade_id", "count"),
    ).reset_index()
    holdings["avg_cost"] = holdings["cost_basis"] / holdings["shares"]

    import shutil, tempfile
    snap = Path(tempfile.gettempdir()) / "ph_daily_prices/"
    if not snap.exists():
        shutil.copy2(PRICES_FILE, snap)
    tickers = holdings["ticker"].astype(str).str.upper().tolist()
    prices = pd.read_parquet(snap, columns=["date", "ticker", "adj_close", "close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices[prices["ticker"].isin(tickers)]
    prices["date"] = pd.to_datetime(prices["date"])
    px = prices["adj_close"].where(prices["adj_close"].notna(), prices["close"])
    latest = prices.assign(px=px).sort_values("date").groupby("ticker").tail(1).set_index("ticker")["px"]
    holdings["ticker"] = holdings["ticker"].astype(str).str.upper()
    holdings["last_close"] = holdings["ticker"].map(latest)
    holdings["market_value"] = holdings["shares"] * holdings["last_close"]
    holdings["unrealized_pl"] = holdings["market_value"] - holdings["cost_basis"]
    holdings["unrealized_pl_pct"] = holdings["unrealized_pl"] / holdings["cost_basis"] * 100
    holdings["weight"] = holdings["market_value"] / holdings["market_value"].sum() * 100

    # Attach sector
    if STOCKS_FILE.exists():
        stocks = pd.read_parquet(STOCKS_FILE)[["ticker", "sector", "industry"]]
        holdings = holdings.merge(stocks, on="ticker", how="left")

    pq.write_table(pa.Table.from_pandas(holdings, preserve_index=False), HOLDINGS_FILE)
    return holdings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Rebuild from trades + latest prices")
    args = parser.parse_args()

    if args.refresh or not HOLDINGS_FILE.exists():
        holdings = rebuild_holdings()
        print("Holdings refreshed from trades.parquet\n")
    else:
        holdings = pd.read_parquet(HOLDINGS_FILE)

    print("=" * 72)
    print("PORTFOLIO SNAPSHOT")
    print("=" * 72)
    cols = ["ticker", "shares", "avg_cost", "cost_basis", "last_close",
            "market_value", "unrealized_pl", "unrealized_pl_pct", "weight"]
    if "sector" in holdings.columns:
        cols.insert(1, "sector")
    print(holdings[cols].round(2).to_string(index=False))

    total_cost = holdings["cost_basis"].sum()
    total_mv = holdings["market_value"].sum()
    total_pl = holdings["unrealized_pl"].sum()
    print("-" * 72)
    print(f"{'TOTAL':12}  cost ${total_cost:,.2f}   MV ${total_mv:,.2f}   "
          f"P&L ${total_pl:+,.2f}  ({total_pl/total_cost*100:+.1f}%)")
    print()

    if "sector" in holdings.columns:
        print("Sector allocation (by market value):")
        sec = holdings.groupby("sector")["market_value"].sum().sort_values(ascending=False)
        for s, v in sec.items():
            print(f"  {s:30} ${v:8.2f}  ({v/total_mv*100:5.1f}%)")
    print()

    # Fundamentals snapshot for holdings
    fund_file = DATA_DIR / "fundamentals.parquet"
    if fund_file.exists():
        fund = pd.read_parquet(fund_file)
        # latest per ticker
        fund = fund.sort_values("as_of_date").groupby("ticker").tail(1)
        # Fresh market cap from PIT panel
        px = pd.read_parquet(DATA_DIR / "daily_mcap.parquet", columns=["ticker", "date", "market_cap"])
        px = px[px["market_cap"].notna()].sort_values("date").groupby("ticker").tail(1)
        daily_mc = px.set_index("ticker")["market_cap"] / 1e9
        fund = fund.set_index("ticker")
        fund["market_cap_b"] = daily_mc.reindex(fund.index).fillna(fund["market_cap_b"])
        fund = fund.reset_index()
        merged = holdings.merge(
            fund[["ticker", "market_cap_b", "total_assets_b", "pb_ratio", "mktcap_to_assets"]],
            on="ticker", how="left"
        )
        print("Fundamentals (P/B & Market Cap / Total Assets):")
        cols = ["ticker", "market_cap_b", "total_assets_b", "pb_ratio", "mktcap_to_assets"]
        print(merged[cols].round(2).to_string(index=False))
        print()


if __name__ == "__main__":
    main()

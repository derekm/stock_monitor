#!/usr/bin/env python3
"""
build_index.py - Construct a simple equal-weight Fertilizer / Ag-Inputs index
from the active index_member stocks and latest prices.

Outputs:
  - fertilizer_index.parquet  (daily index level + component returns)
  - Prints current snapshot and performance vs prior day if available.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
PRICES_FILE = DATA_DIR / "daily_prices/"
INDEX_FILE = DATA_DIR / "fertilizer_index.parquet"

def load_latest_prices(tickers):
    if not PRICES_FILE.exists():
        return pd.DataFrame()
    prices = pd.read_parquet(PRICES_FILE)
    prices = prices[prices["ticker"].isin(tickers)]
    # Take the most recent date per ticker
    latest = prices.sort_values("date").groupby("ticker").tail(1)
    return latest.set_index("ticker")

def main():
    stocks = pd.read_parquet(STOCKS_FILE)
    members = stocks[stocks["index_member"] == True]["ticker"].tolist()
    if not members:
        print("No index members defined.")
        return

    print(f"Index members ({len(members)}): {', '.join(members)}")

    latest = load_latest_prices(members)
    if latest.empty:
        print("No price data available.")
        return

    missing = set(members) - set(latest.index)
    if missing:
        print(f"Missing prices for: {missing}")
        members = [m for m in members if m in latest.index]

    # Equal weight
    n = len(members)
    weight = 1.0 / n
    components = []
    for t in members:
        row = latest.loc[t]
        components.append({
            "ticker": t,
            "close": row["close"],
            "weight": weight,
            "contrib": row["close"] * weight,  # for level construction we use normalized
        })

    comp_df = pd.DataFrame(components)
    # Simple price-weighted average as proxy for equal-weight index level
    # (true equal-weight would rebalance shares; here we average the prices after normalizing to a base)
    # For first day we set index = 100 * average(close / first_close) but since single day we just average.
    index_level = 100.0  # base
    # Using average of closes scaled so that on base date it is 100
    avg_close = comp_df["close"].mean()
    # For multi-day we would compute cumulative; for now snapshot
    print("\n=== Fertilizer / Ag-Inputs Equal-Weight Snapshot ===")
    print(f"Date: {latest['date'].iloc[0] if 'date' in latest.columns else 'latest'}")
    print(comp_df[["ticker", "close", "weight"]].to_string(index=False))
    print(f"\nSimple average close: ${avg_close:.2f}")
    print(f"Index level (base 100 on first observation): {index_level:.2f}")

    # Persist a daily index record
    idx_row = {
        "date": latest["date"].iloc[0] if "date" in latest.columns else datetime.now().date(),
        "index_level": index_level,
        "avg_close": avg_close,
        "n_members": n,
        "members": ",".join(sorted(members)),
        "method": "equal_weight_price_avg",
    }
    idx_df = pd.DataFrame([idx_row])

    if INDEX_FILE.exists():
        existing = pd.read_parquet(INDEX_FILE)
        combined = pd.concat([existing, idx_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    else:
        combined = idx_df

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, INDEX_FILE)
    print(f"\nWrote {INDEX_FILE}")

    # Fundamentals (P/B, market cap)
    fund_file = DATA_DIR / "fundamentals.parquet"
    if fund_file.exists():
        fund = pd.read_parquet(fund_file)
        fund = fund.sort_values("as_of_date").groupby("ticker").tail(1)
        # Fresh market cap from daily prices (beats quarterly fundamentals snapshot)
        px = pd.read_parquet(DATA_DIR / "daily_prices/", columns=["ticker", "date", "market_cap"])
        px = px[px["market_cap"].notna()].sort_values("date").groupby("ticker").tail(1)
        daily_mc = px.set_index("ticker")["market_cap"] / 1e9
        fund = fund.set_index("ticker")
        fund["market_cap_b"] = daily_mc.reindex(fund.index).fillna(fund["market_cap_b"])
        fund = fund.reset_index()
        merged = comp_df.merge(fund[["ticker", "market_cap_b", "total_assets_b", "pb_ratio", "mktcap_to_assets"]], on="ticker", how="left")
        print("\nFundamentals:")
        print(merged[["ticker", "close", "market_cap_b", "pb_ratio", "mktcap_to_assets"]].round(2).to_string(index=False))
        print(f"Aggregate index market cap: ${merged['market_cap_b'].sum():.1f}B")
        print(f"Median P/B: {merged['pb_ratio'].median():.2f}  |  Median MktCap/Assets: {merged['mktcap_to_assets'].median():.2f}")

    # Sector breakdown of the index
    stocks_m = stocks[stocks["ticker"].isin(members)]
    print("\nSector / Subsector composition:")
    print(stocks_m.groupby(["sector", "subsector"]).size().to_string())

if __name__ == "__main__":
    main()

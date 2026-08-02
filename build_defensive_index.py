#!/usr/bin/env python3
"""
build_defensive_index.py - Equal-weight Defensive / Value index snapshot.

Members = all tickers with defensive_value_index=True in monitored_stocks.parquet
(Staples, Healthcare/Pharma, Telecom/Utilities, select Industrials).

Outputs defensive_value_index.parquet and prints P/B, EV/EBITDA, market-cap context.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
FUND_FILE = DATA_DIR / "fundamentals.parquet"
INDEX_FILE = DATA_DIR / "defensive_value_index.parquet"


def main():
    stocks = pd.read_parquet(STOCKS_FILE)
    if "defensive_value_index" not in stocks.columns:
        print("No defensive_value_index flag found.")
        return
    members = stocks[stocks["defensive_value_index"] == True]["ticker"].tolist()
    if not members:
        print("No defensive-value index members.")
        return

    prices = pd.read_parquet(PRICES_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    latest = prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

    fund = pd.read_parquet(FUND_FILE) if FUND_FILE.exists() else pd.DataFrame()
    if not fund.empty:
        fund = fund.sort_values("as_of_date").groupby("ticker").tail(1).set_index("ticker")

    rows = []
    for t in members:
        if t not in latest.index:
            continue
        r = {
            "ticker": t,
            "close": float(latest.loc[t, "close"]),
            "sector": stocks.loc[stocks.ticker == t, "sector"].iloc[0],
        }
        if not fund.empty and t in fund.index:
            r["market_cap_b"] = fund.loc[t, "market_cap_b"]
            r["pb_ratio"] = fund.loc[t, "pb_ratio"]
            r["ev_ebitda"] = fund.loc[t, "ev_ebitda"] if "ev_ebitda" in fund.columns else None
            r["mktcap_to_assets"] = fund.loc[t, "mktcap_to_assets"]
        rows.append(r)

    comp = pd.DataFrame(rows)
    n = len(comp)
    comp["weight"] = 1.0 / n

    print("=" * 78)
    print(f"DEFENSIVE / VALUE INDEX  ({n} equal-weight members)")
    print("=" * 78)
    cols = [c for c in ["ticker", "sector", "close", "market_cap_b", "pb_ratio", "ev_ebitda", "weight"] if c in comp.columns]
    print(comp.sort_values(["sector", "ticker"])[cols].round(2).to_string(index=False))

    print(f"\nSimple average close: ${comp['close'].mean():.2f}")
    if "market_cap_b" in comp.columns:
        print(f"Aggregate market cap: ${comp['market_cap_b'].sum():.0f}B")
        print(f"Median P/B: {comp['pb_ratio'].median():.2f}  |  Median EV/EBITDA: {comp['ev_ebitda'].median():.1f}")

    print("\nBy sector:")
    print(comp.groupby("sector").size().to_string())

    # Persist snapshot
    idx_row = {
        "date": latest["date"].iloc[0] if "date" in latest.columns else pd.Timestamp.now().normalize(),
        "index_level": 100.0,
        "avg_close": comp["close"].mean(),
        "n_members": n,
        "members": ",".join(sorted(comp["ticker"])),
        "method": "equal_weight",
        "median_pb": float(comp["pb_ratio"].median()) if "pb_ratio" in comp.columns else None,
        "median_ev_ebitda": float(comp["ev_ebitda"].median()) if "ev_ebitda" in comp.columns else None,
    }
    idx_df = pd.DataFrame([idx_row])
    if INDEX_FILE.exists():
        existing = pd.read_parquet(INDEX_FILE)
        combined = pd.concat([existing, idx_df], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    else:
        combined = idx_df
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), INDEX_FILE)
    print(f"\nWrote {INDEX_FILE}")


if __name__ == "__main__":
    main()

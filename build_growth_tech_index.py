#!/usr/bin/env python3
"""
build_growth_tech_index.py — Equal-weight higher-risk Growth / Tech index.

Sleeves (growth_sleeve on monitored_stocks):
  growth_ai       — SMCI, NVDA, AMD, PLTR, CRWD
  quality_growth  — MSFT, GOOGL
  emerging_growth — TSLA, ENPH, SEDG, REGN, XBI
  cyclical        — BA, CAT, SCHW
  thematic        — ARKK, QQQ, VUG  (small satellite slice)

Members = growth_tech_index=True. Designed as a 4th sleeve alongside
fertilizer, defensive_value, and personal portfolio — higher vol, capped sizing.
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
STOCKS = DATA_DIR / "monitored_stocks.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "growth_tech_index.parquet"
LEVELS = DATA_DIR / "growth_tech_index_levels.parquet"


def main():
    stocks = pd.read_parquet(STOCKS)
    if "growth_tech_index" not in stocks.columns:
        print("No growth_tech_index flag — run membership update first.")
        return
    members = stocks[stocks["growth_tech_index"] == True].copy()
    tickers = members["ticker"].tolist()
    if not tickers:
        print("No growth-tech members.")
        return

    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    latest = prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

    fund = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    if not fund.empty and "as_of_date" in fund.columns:
        fund = fund.sort_values("as_of_date").groupby("ticker").tail(1).set_index("ticker")
    elif not fund.empty:
        fund = fund.groupby("ticker").tail(1).set_index("ticker")

    rows = []
    for t in tickers:
        if t not in latest.index:
            continue
        px = float(latest.loc[t, "close"])
        meta = members[members["ticker"] == t].iloc[0]
        fr = fund.loc[t] if (not fund.empty and t in fund.index) else None
        rows.append({
            "ticker": t,
            "name": meta.get("name"),
            "sector": meta.get("sector"),
            "growth_sleeve": meta.get("growth_sleeve"),
            "notes": meta.get("notes"),
            "last_close": px,
            "as_of": latest.loc[t, "date"],
            "pb_ratio": float(fr["pb_ratio"]) if fr is not None and "pb_ratio" in fr and pd.notna(fr["pb_ratio"]) else None,
            "ev_ebitda": float(fr["ev_ebitda"]) if fr is not None and "ev_ebitda" in fr and pd.notna(fr["ev_ebitda"]) else None,
            "weight_ew": 1.0 / len(tickers),
            "risk_bucket": "higher",
        })

    snap = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(snap, preserve_index=False), OUT)
    print(f"Growth/Tech index snapshot → {OUT} ({len(snap)} members)")
    print(snap.groupby("growth_sleeve")["ticker"].apply(list).to_string())
    print(snap[["ticker", "growth_sleeve", "last_close", "ev_ebitda", "weight_ew"]].to_string(index=False))

    # Equal-weight index levels (base 100)
    wide = prices[prices["ticker"].isin(tickers)].pivot_table(
        index="date", columns="ticker", values="close"
    ).sort_index().ffill()
    wide = wide.dropna(how="all")
    if len(wide) >= 2:
        rets = wide.pct_change()
        ew = rets.mean(axis=1).fillna(0)
        level = (1 + ew).cumprod() * 100
        level = level / level.iloc[0] * 100
        lv = level.rename("index_level").reset_index()
        lv["index_name"] = "growth_tech"
        pq.write_table(pa.Table.from_pandas(lv, preserve_index=False), LEVELS)
        print(f"Levels → {LEVELS}  last={level.iloc[-1]:.2f}  "
              f"({level.index[0].date()} → {level.index[-1].date()})")


if __name__ == "__main__":
    main()

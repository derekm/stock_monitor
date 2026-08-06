#!/usr/bin/env python3
"""
economic_calendar.py — Trading-day, options-expiry, FOMC and earnings-event
calendar for the monitored universe.

Why it exists: the architecture TODO "economic calendars" — regime-aware
scheduling and earnings-adjacent analytics need to know when events land.
Zero new dependencies: trading days come from daily_prices.parquet (the
actual market calendar the repo already trades on), quarterly expiries are
computed (3rd Friday of Mar/Jun/Sep/Dec), FOMC meetings are curated (the Fed
publishes the schedule years ahead — update macro_events.csv annually), and
earnings dates come from earnings_calendar.parquet.

Output: economic_calendar.csv — one row per (date, event_type) with a
days-until flag for near-term events.

Usage:
    python economic_calendar.py [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import calendar
from datetime import date, timedelta

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"
EARNINGS = DATA_DIR / "earnings_calendar.parquet"
MACRO = DATA_DIR / "macro_events.csv"
OUT = DATA_DIR / "economic_calendar.csv"

# Curated FOMC meeting dates (Fed publishes years ahead; update annually).
# Format: date,event_type,label,source
FOMC_DEFAULT = """2025-01-29,FOMC,FOMC decision,federalreserve.gov
2025-03-19,FOMC,FOMC decision,federalreserve.gov
2025-05-07,FOMC,FOMC decision,federalreserve.gov
2025-06-18,FOMC,FOMC decision,federalreserve.gov
2025-07-30,FOMC,FOMC decision,federalreserve.gov
2025-09-17,FOMC,FOMC decision,federalreserve.gov
2025-10-29,FOMC,FOMC decision,federalreserve.gov
2025-12-10,FOMC,FOMC decision,federalreserve.gov
2026-01-28,FOMC,FOMC decision,federalreserve.gov
2026-03-18,FOMC,FOMC decision,federalreserve.gov
2026-04-29,FOMC,FOMC decision,federalreserve.gov
2026-06-17,FOMC,FOMC decision,federalreserve.gov
2026-07-29,FOMC,FOMC decision,federalreserve.gov
2026-09-16,FOMC,FOMC decision,federalreserve.gov
2026-10-28,FOMC,FOMC decision,federalreserve.gov
2026-12-09,FOMC,FOMC decision,federalreserve.gov
2027-01-27,FOMC,FOMC decision,federalreserve.gov
2027-03-17,FOMC,FOMC decision,federalreserve.gov
2027-04-28,FOMC,FOMC decision,federalreserve.gov
2027-06-16,FOMC,FOMC decision,federalreserve.gov
2027-07-28,FOMC,FOMC decision,federalreserve.gov
2027-09-15,FOMC,FOMC decision,federalreserve.gov
2027-10-27,FOMC,FOMC decision,federalreserve.gov
2027-12-08,FOMC,FOMC decision,federalreserve.gov
"""


def quarterly_expiries(years: range) -> list[date]:
    """Third Friday of Mar/Jun/Sep/Dec — standard quarterly options expiry."""
    out = []
    for y in years:
        for month in (3, 6, 9, 12):
            cal = calendar.monthcalendar(y, month)
            fridays = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY] != 0]
            if len(fridays) >= 3:
                out.append(date(y, month, fridays[2]))
    return out


def trading_days(prices: pd.DataFrame) -> set[date]:
    """Actual trading days from the price spine (the repo's market calendar)."""
    days = set()
    if "date" in prices.columns:
        days = {d.date() if hasattr(d, "date") else d
                for d in pd.to_datetime(prices["date"]).unique()}
    return days


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    today = date.today()
    prices = pd.read_parquet(PRICES, columns=["date"]) if PRICES.exists() else pd.DataFrame()
    tdays = trading_days(prices)
    lo = min(tdays) if tdays else today - timedelta(days=365)
    hi = max(tdays) if tdays else today + timedelta(days=365)
    years = range(lo.year - 1, hi.year + 2)

    rows: list[dict] = []
    # 1) quarterly options expiries
    for d in quarterly_expiries(years):
        if lo <= d <= hi + timedelta(days=400):
            rows.append({"date": d, "event_type": "option_expiry",
                         "label": "Quarterly expiry", "source": "computed (3rd Fri)"})
    # 2) FOMC schedule — curated file if present, else embedded default
    macro_file = MACRO if MACRO.exists() else None
    if macro_file:
        m = pd.read_csv(macro_file)
        for _, r in m.iterrows():
            rows.append({"date": pd.to_datetime(r["date"]).date(),
                         "event_type": str(r.get("event_type", "macro")),
                         "label": str(r.get("label", "")),
                         "source": str(r.get("source", "macro_events.csv"))})
    else:
        for line in FOMC_DEFAULT.strip().splitlines():
            d, et, lab, src = line.split(",")
            rows.append({"date": pd.to_datetime(d).date(),
                         "event_type": et, "label": lab, "source": src})
    # 3) earnings events per ticker (near-term only)
    if EARNINGS.exists():
        e = pd.read_parquet(EARNINGS)
        if "earnings_date" in e.columns and "ticker" in e.columns:
            e["earnings_date"] = pd.to_datetime(e["earnings_date"]).dt.date
            near = e[e["earnings_date"] >= today - timedelta(days=7)]
            for _, r in near.head(2000).iterrows():
                rows.append({"date": r["earnings_date"], "event_type": "earnings",
                             "label": f"{r['ticker']} earnings",
                             "source": "earnings_calendar.parquet"})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["days_until"] = (df["date"] - pd.Timestamp(today)).dt.days
    df = df.sort_values(["date", "event_type"]).drop_duplicates(["date", "event_type", "label"])
    df["is_trading_day"] = df["date"].dt.date.isin(tdays)

    print(f"=== Economic calendar ({len(df)} events, {lo} → {hi}) ===")
    print(f"trading days in spine: {len(tdays)}")
    print(df[df["days_until"].between(-7, 30)]
          [["date", "event_type", "label", "days_until", "is_trading_day"]]
          .to_string(index=False))
    if args.save:
        df.to_csv(OUT, index=False)
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

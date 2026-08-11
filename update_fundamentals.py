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
        "ticker", "as_of_date", "market_cap", "market_cap_b", "shares_outstanding",
        "total_assets", "total_assets_b", "pb_ratio", "mktcap_to_assets", "source",
        "notes", "last_updated"
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

    # shares_outstanding: direct count if the price at as_of is available,
    # else derived from market cap (mcap / close). Prefer an explicit count.
    shares = None
    px = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["ticker", "date", "close"])
    px = px[px["ticker"] == t]
    px["date"] = pd.to_datetime(px["date"])
    row_close = px[px["date"] <= pd.Timestamp(as_of)]
    if len(row_close):
        shares = (mcap_b * 1e9) / float(row_close["close"].iloc[-1])

    row = {
        "ticker": t,
        "as_of_date": as_of,
        "market_cap": int(mcap_b * 1e9),
        "market_cap_b": mcap_b,
        "shares_outstanding": shares,
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
    # shares_outstanding: explicit column if provided, else derive from mcap/close
    if "shares_outstanding" not in csv_df.columns:
        px = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["ticker", "date", "close"])
        px["date"] = pd.to_datetime(px["date"])
        def _shares(t, asof):
            sub = px[(px["ticker"] == t) & (px["date"] <= pd.Timestamp(asof))]
            if not len(sub):
                return None
            mcap_b = csv_df.loc[csv_df["ticker"] == t, "market_cap_b"].iloc[0]
            return (float(mcap_b) * 1e9) / float(sub["close"].iloc[-1])
        csv_df["shares_outstanding"] = [None] * len(csv_df)
        for i, r in csv_df.iterrows():
            csv_df.at[i, "shares_outstanding"] = _shares(r["ticker"], r["as_of_date"])
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

    # Source-priority: don't overwrite EDGAR rows with same-day yfinance info
    edgar_keys = set(
        zip(
            existing.loc[existing.get("source") == "edgar", "ticker"],
            existing.loc[existing.get("source") == "edgar", "as_of_date"],
        )
    )
    new_df = new_df[
        ~new_df.apply(lambda r: (r["ticker"], r["as_of_date"]) in edgar_keys, axis=1)
    ]
    if len(new_df) != len(rows):
        print(f"  (skipped {len(rows) - len(new_df)} rows whose date is covered by EDGAR)")
        if new_df.empty:
            save(existing)
            print("  nothing new to add (all rows already EDGAR)")
            return

    keys = set(zip(new_df["ticker"], new_df["as_of_date"]))
    existing = existing[~existing.apply(lambda r: (r["ticker"], r["as_of_date"]) in keys, axis=1)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    save(combined)


def cmd_fetch_history(args):
    """Real point-in-time fundamentals: quarterly statements from yfinance.

    For each ticker pulls quarterly income statement + balance sheet and
    computes as-of-quarter-end: ROE (TTM NI / equity), ROIC (TTM NOPAT /
    invested capital), D/E, EV/EBITDA (mktcap + debt - cash)/TTM EBITDA,
    P/B (mktcap / equity), MktCap/Assets. Market cap = price × shares at the
    quarter end (from daily_prices.parquet adj_close, last price <= qend).

    These are REAL dated rows (source=yfinance_history), replacing the
    synthetic mean-reverting backfill that fundamentals_history.py generates.
    """
    import yfinance as yf

    if STOCKS_FILE.exists():
        tickers = sorted(pd.read_parquet(STOCKS_FILE)["ticker"].astype(str).str.upper().unique().tolist())
    else:
        tickers = sorted(load()["ticker"].unique().tolist())
    skip = {".", "^", "=", "-", ":"}
    tickers = [t for t in tickers if not any(s in t for s in skip)]
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.max_tickers:
        tickers = tickers[: args.max_tickers]

    # price series: ticker -> DataFrame(date, close=adj_close) for mktcap at qend
    try:
        from analytics_common import load_adj_prices_pandas
        prices = load_adj_prices_pandas(tickers=tickers)
        px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}
    except Exception as e:  # noqa: BLE001
        print(f"  !! price load failed: {e}; market-cap-based rows will be None")
        px = {}

    new_rows: list[dict] = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            inc = tk.get_income_stmt(freq="quarterly")
            bal = tk.get_balance_sheet(freq="quarterly")
            if inc is None or bal is None or inc.empty or bal.empty:
                print(f"  !! {t}: no statements")
                continue
            # quarter-end dates = columns (tz-aware); use both frames' union
            dates = sorted(set(inc.columns) | set(bal.columns))
            for d in dates:
                qend = d.date() if hasattr(d, "date") else pd.Timestamp(d).date()
                # TTM income over 4 quarters ending at d
                qi = inc.loc[:, inc.columns <= d]
                qb = bal.loc[:, bal.columns <= d]
                if qi.shape[1] == 0 or qb.shape[1] == 0:
                    continue
                def ttm(row_name: str) -> float | None:
                    if row_name not in qi.index:
                        return None
                    s = qi.loc[row_name].dropna().tail(4)
                    return float(s.sum()) if len(s) else None
                def balv(row_name: str) -> float | None:
                    if row_name not in qb.index:
                        return None
                    s = qb.loc[row_name].dropna()
                    return float(s.iloc[-1]) if len(s) else None
                ni = ttm("NetIncomeCommonStockholders")
                oi = ttm("OperatingIncome")
                ebitda = ttm("EBITDA")
                equity = balv("StockholdersEquity")
                total_assets = balv("TotalAssets")
                debt = balv("TotalDebt")
                cash = balv("CashAndCashEquivalents")
                shares = balv("OrdinarySharesNumber")
                # market cap at qend from price × shares (last close <= qend)
                mcap = None
                if t in px and shares:
                    p = px[t]
                    avail = p[p.index <= pd.Timestamp(qend)]
                    if len(avail):
                        mcap = float(avail.iloc[-1]) * shares
                mcap_b = mcap / 1e9 if mcap else None
                roe = ni / equity if ni and equity else None
                nopat = oi * 0.75 if oi else None  # ~25% effective tax proxy
                invested = balv("InvestedCapital")
                roic = nopat / invested if nopat and invested else None
                de = debt / equity if debt and equity else None
                ev = (mcap + debt - cash) if mcap and debt and cash else None
                ev_ebitda = ev / ebitda if ev and ebitda else None
                pb = mcap / equity if mcap and equity else None
                mca = mcap / total_assets if mcap and total_assets else None
                new_rows.append({
                    "ticker": t,
                    "as_of_date": qend,
                    "market_cap": int(mcap) if mcap else None,
                    "market_cap_b": round(mcap_b, 2) if mcap_b else None,
                    "total_assets": int(total_assets) if total_assets else None,
                    "total_assets_b": round(total_assets / 1e9, 2) if total_assets else None,
                    "pb_ratio": round(pb, 3) if pb else None,
                    "mktcap_to_assets": round(mca, 3) if mca else None,
                    "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
                    "roe": round(roe, 4) if roe else None,
                    "roic": round(roic, 4) if roic else None,
                    "debt_to_equity": round(de, 3) if de else None,
                    "source": "yfinance_history",
                    "notes": "real quarterly statements (TTM income)",
                    "last_updated": pd.Timestamp.now(),
                })
            print(f"  {t}: {len([r for r in new_rows if r['ticker'] == t])} quarters")
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: {e}")
    if not new_rows:
        print("No rows fetched.")
        return
    new_df = pd.DataFrame(new_rows)
    # drop any synthetic backfill rows for these tickers (real data replaces noise)
    existing = load()
    real_tickers = set(new_df["ticker"])
    existing = existing[
        ~(
            existing["ticker"].isin(real_tickers)
            & (existing.get("source") == "fundamentals_history_backfill")
        )
    ]
    # ── Source-priority guard: EDGAR is the gold standard (point-in-time,
    # as-reported XBRL). Never let a yfinance row overwrite an existing EDGAR
    # row for the same (ticker, as_of_date). Source priority:
    #   edgar > manual > yfinance_history > yfinance > fundamentals_history_backfill
    edgar_keys = set(
        zip(
            existing.loc[existing.get("source") == "edgar", "ticker"],
            existing.loc[existing.get("source") == "edgar", "as_of_date"],
        )
    )
    new_df = new_df[
        ~new_df.apply(lambda r: (r["ticker"], r["as_of_date"]) in edgar_keys, axis=1)
    ]
    if len(new_df) != len(new_rows):
        print(f"  (skipped {len(new_rows) - len(new_df)} rows whose quarter is covered by EDGAR)")
        if new_df.empty:
            save(existing)
            print("  nothing new to add (all quarters already EDGAR)")
            return
    keys = set(zip(new_df["ticker"], new_df["as_of_date"]))
    existing = existing[~existing.apply(lambda r: (r["ticker"], r["as_of_date"]) in keys, axis=1)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    save(combined)
    print(f"Fetched real history: {len(new_df)} rows for {new_df['ticker'].nunique()} tickers")


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

    p = sub.add_parser("fetch-history")
    p.add_argument("--tickers", help="Comma-separated subset")
    p.add_argument("--max-tickers", type=int, default=None, help="Cap batch size")
    p.set_defaults(func=cmd_fetch_history)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

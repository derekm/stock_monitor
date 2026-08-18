#!/usr/bin/env python3
"""
update_fundamentals.py — Update P/B, market cap, total assets, and quarterly fundamentals.

Commands:
  fetch          — snapshot from yfinance .info (daily)
  fetch-history  — REAL point-in-time quarterly fundamentals from yfinance quarterly statements
  manual         — manual entry
  from-csv       — bulk load from CSV
  show           — inspect latest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
MONITORED = DATA_DIR / "monitored_stocks.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"

# Source priority (higher = better) — used by preferred_metrics.py
SOURCE_RANK = {
    "edgar": 100,
    "manual": 80,
    "yfinance_history": 60,
    "polygon_financials": 55,
    "yfinance": 40,
    "fundamentals_history_backfill": 10,
}


def _as_date(x):
    if isinstance(x, date) and not isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, str):
        return pd.Timestamp(x).date()
    if pd.isna(x):
        return pd.NaT
    return pd.Timestamp(x).date()


def load() -> pd.DataFrame:
    if FUND.exists():
        df = pd.read_parquet(FUND)
        if "as_of_date" in df.columns:
            df["as_of_date"] = df["as_of_date"].map(_as_date)
        return df
    return pd.DataFrame()


def save(df: pd.DataFrame) -> None:
    df = df.copy()
    if "as_of_date" in df.columns:
        df["as_of_date"] = df["as_of_date"].map(_as_date)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), FUND)


def universe_tickers() -> list[str] | None:
    """Get universe tickers from daily_prices (NOT monitored_stocks)."""
    if PRICES.exists():
        df = pd.read_parquet(PRICES, columns=["ticker"])
        return df["ticker"].dropna().unique().tolist()
    return None


def cmd_show(args):
    df = load()
    if df.empty:
        print("No data")
        return
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
    print(df.sort_values("as_of_date", ascending=False).head(20).to_string(index=False))


def cmd_manual(args):
    ticker = args.ticker.upper()
    as_of = _as_date(args.as_of) if args.as_of else date.today()
    row = {
        "ticker": ticker,
        "as_of_date": as_of,
        "market_cap_b": args.market_cap_b,
        "total_assets_b": args.total_assets_b,
        "market_cap": int(args.market_cap_b * 1e9),
        "total_assets": int(args.total_assets_b * 1e9),
        "pb_ratio": args.pb,
        "ev_ebitda": args.ev_ebitda,
        "source": "manual",
        "notes": args.notes,
        "last_updated": pd.Timestamp.now(),
    }
    df = load()
    df["as_of_date"] = df["as_of_date"].map(_as_date)
    idx = df.index[(df["ticker"] == ticker) & (df["as_of_date"] == as_of)].tolist()
    if idx:
        df.loc[idx[0]] = row
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    print(f"Manual entry for {ticker} @ {as_of} saved")


def cmd_from_csv(args):
    df_csv = pd.read_csv(args.csv)
    df_csv.columns = [c.strip().lower().replace(" ", "_") for c in df_csv.columns]
    required = {"ticker", "market_cap_b", "total_assets_b"}
    if not required.issubset(df_csv.columns):
        raise ValueError(f"CSV must have columns: {required}")
    df_csv["ticker"] = df_csv["ticker"].str.upper()
    if "as_of_date" in df_csv.columns:
        df_csv["as_of_date"] = df_csv["as_of_date"].map(_as_date)
    else:
        df_csv["as_of_date"] = date.today()
    df_csv["market_cap"] = (df_csv["market_cap_b"] * 1e9).astype(int)
    df_csv["total_assets"] = (df_csv["total_assets_b"] * 1e9).astype(int)
    df_csv["source"] = "from_csv"
    df_csv["last_updated"] = pd.Timestamp.now()
    existing = load()
    existing["as_of_date"] = existing["as_of_date"].map(_as_date)
    combined = pd.concat([existing, df_csv], ignore_index=True)
    save(combined)
    print(f"Loaded {len(df_csv)} rows from {args.csv}")


def cmd_fetch(args):
    """Snapshot from yfinance .info — daily, not historical."""
    import yfinance as yf

    tickers = universe_tickers() or sorted(load()["ticker"].unique().tolist())
    skip = {".", "^", "=", "-", ":"}
    tickers = [t for t in tickers if not any(s in t for s in skip)]
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"Fetching snapshot for {len(tickers)} tickers...")

    new_rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.get_info()
            mcap = info.get("marketCap")
            assets = info.get("totalAssets")
            if mcap and assets:
                new_rows.append({
                    "ticker": t,
                    "as_of_date": date.today(),
                    "market_cap": int(mcap),
                    "market_cap_b": round(mcap / 1e9, 2),
                    "total_assets": int(assets),
                    "total_assets_b": round(assets / 1e9, 2),
                    "pb_ratio": info.get("priceToBook"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "roe": info.get("returnOnEquity"),
                    "roic": info.get("returnOnInvestedCapital"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "interest_coverage": None,
                    "earnings_stability": None,
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "source": "yfinance",
                    "notes": "yfinance .info snapshot",
                    "last_updated": pd.Timestamp.now(),
                })
                print(f"  {t}: mcap={mcap/1e9:.2f}B, assets={assets/1e9:.2f}B")
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: {e}")

    if not new_rows:
        print("No rows fetched.")
        return

    new_df = pd.DataFrame(new_rows)
    new_df["as_of_date"] = new_df["as_of_date"].map(_as_date)
    existing = load()
    existing["as_of_date"] = existing["as_of_date"].map(_as_date)

    # Merge: upsert by (ticker, as_of_date)
    idx = ["ticker", "as_of_date"]
    ex = existing.set_index(idx)
    nd = new_df.set_index(idx)
    overlap = ex.index.intersection(nd.index)
    if len(overlap):
        for c in new_df.columns:
            if c not in ex.columns:
                continue
            missing = ex.loc[overlap, c].isna()
            if missing.any():
                ex.loc[overlap, c] = ex.loc[overlap, c].fillna(nd.loc[overlap, c])
    brand_new = new_df[~new_df.set_index(idx).index.isin(ex.index)].copy()
    combined = pd.concat([ex.reset_index(), brand_new], ignore_index=True) if len(brand_new) else ex.reset_index()
    save(combined)
    print(f"Fetch done: {len(new_df)} rows fetched")


def cmd_fetch_history(args):
    """Real point-in-time fundamentals: quarterly AND annual statements from yfinance.

    For each ticker pulls quarterly + annual income statement, balance sheet, cash flow
    and computes as-of-period-end: ROE (TTM NI / equity), ROIC (TTM NOPAT /
    invested capital), D/E, EV/EBITDA (mktcap + debt - cash)/TTM EBITDA,
    P/B (mktcap / equity), MktCap/Assets. Market cap = price × shares at the
    period end (from daily_prices.parquet adj_close, last price <= period end).

    Also extracts: TotalRevenue, FreeCashFlow, CapitalExpenditure for
    Damodaran life cycle / fair multiple calculations.

    These are REAL dated rows (source=yfinance_history), replacing the
    synthetic mean-reverting backfill that fundamentals_history.py generates.
    """
    import yfinance as yf

    tickers = universe_tickers() or sorted(load()["ticker"].unique().tolist())
    skip = {".", "^", "=", "-", ":"}
    tickers = [t for t in tickers if not any(s in t for s in skip)]
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"fetch-history universe: {len(tickers)} tickers")

    # price series: ticker -> DataFrame(date, close=adj_close) for mktcap at period end
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
            
            # Fetch BOTH quarterly (new API) and annual (old API) for maximum history
            inc_q = tk.get_income_stmt(freq="quarterly")
            bal_q = tk.get_balance_sheet(freq="quarterly")
            cf_q = tk.get_cashflow(freq="quarterly")
            
            inc_a = tk.financials  # annual
            bal_a = tk.balance_sheet
            cf_a = tk.cashflow
            
            if inc_q is None or bal_q is None or inc_q.empty or bal_q.empty:
                print(f"  !! {t}: no quarterly statements")
                continue

            # Build unified frames: combine quarterly + annual
            # Quarterly columns
            dates_q = sorted(set(inc_q.columns) | set(bal_q.columns) | (set(cf_q.columns) if cf_q is not None else set()))
            # Annual columns
            dates_a = sorted(set(inc_a.columns) | set(bal_a.columns) | (set(cf_a.columns) if cf_a is not None else set()))
            all_dates = sorted(set(dates_q) | set(dates_a))
            
            for d in all_dates:
                pend = d.date() if hasattr(d, "date") else pd.Timestamp(d).date()
                
                # Determine which frame to use
                use_quarterly = d in inc_q.columns
                use_annual = d in inc_a.columns
                
                # TTM requires 4 quarters - only works with quarterly data
                if use_quarterly:
                    qi = inc_q.loc[:, inc_q.columns <= d]
                    qb = bal_q.loc[:, bal_q.columns <= d]
                    qc = cf_q.loc[:, cf_q.columns <= d] if cf_q is not None else None
                    if qi.shape[1] == 0 or qb.shape[1] == 0:
                        continue
                elif use_annual:
                    # For annual, we don't do TTM - just use the single year
                    qi = inc_a.loc[:, inc_a.columns <= d]
                    qb = bal_a.loc[:, bal_a.columns <= d]
                    qc = cf_a.loc[:, cf_a.columns <= d] if cf_a is not None else None
                    if qi.shape[1] == 0 or qb.shape[1] == 0:
                        continue
                else:
                    continue
                
                def ttm(row_name: str) -> float | None:
                    if row_name not in qi.index:
                        return None
                    s = qi.loc[row_name].dropna().tail(4)
                    return float(s.sum()) if len(s) else None
                
                def single(row_name: str) -> float | None:
                    """Get single period value (for annual data)"""
                    if row_name not in qi.index:
                        return None
                    s = qi.loc[row_name].dropna()
                    return float(s.iloc[0]) if len(s) else None
                
                def balv(row_name: str) -> float | None:
                    if use_quarterly:
                        # Use quarterly balance sheet
                        frame = qb
                    else:
                        # Use annual balance sheet
                        frame = qb
                    if row_name not in frame.index:
                        return None
                    s = frame.loc[row_name].dropna()
                    return float(s.iloc[0]) if len(s) else None
                
                def cfv(row_name: str) -> float | None:
                    if qc is None or row_name not in qc.index:
                        return None
                    s = qc.loc[row_name].dropna()
                    if use_quarterly:
                        return float(s.tail(4).sum()) if len(s) else None
                    else:
                        return float(s.iloc[0]) if len(s) else None
                
                # For quarterly: TTM. For annual: single period.
                if use_quarterly:
                    ni = ttm("NetIncomeCommonStockholders")
                    oi = ttm("OperatingIncome")
                    ebitda = ttm("EBITDA")
                    revenue = ttm("TotalRevenue")
                    fcf = cfv("FreeCashFlow")
                    capex = cfv("CapitalExpenditure")
                else:
                    ni = single("NetIncomeCommonStockholders")
                    oi = single("OperatingIncome")
                    ebitda = single("EBITDA")
                    revenue = single("TotalRevenue")
                    fcf = cfv("FreeCashFlow")
                    capex = cfv("CapitalExpenditure")
                
                equity = balv("StockholdersEquity")
                total_assets = balv("TotalAssets")
                debt = balv("TotalDebt")
                cash = balv("CashAndCashEquivalents")
                shares = balv("OrdinarySharesNumber")
                
                # market cap at period end from price × shares (last close <= period end)
                mcap = None
                if t in px and shares:
                    p = px[t]
                    avail = p[p.index <= pd.Timestamp(pend)]
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
                # New Damodaran fields
                rev_growth = None  # will compute later per ticker
                fcf_margin = fcf / revenue if fcf and revenue else None
                reinvestment_rate = capex / oi if capex and oi else None
                new_rows.append({
                    "ticker": t,
                    "as_of_date": pend,
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
                    "shares_outstanding": int(shares) if shares else None,
                    "revenue_quarterly": int(revenue) if revenue else None,
                    "free_cash_flow": int(fcf) if fcf else None,
                    "capital_expenditure_ttm": int(capex) if capex else None,
                    "fcf_margin": round(fcf_margin, 4) if fcf_margin else None,
                    "reinvestment_rate": round(reinvestment_rate, 4) if reinvestment_rate else None,
                    "source": "yfinance_history",
                    "notes": "real quarterly/annual statements",
                    "last_updated": pd.Timestamp.now(),
                })
            # Count real quarters (not annual)
            real_q = len([r for r in new_rows if r['ticker'] == t and use_quarterly]) if new_rows else 0
            print(f"  {t}: {len([r for r in new_rows if r['ticker'] == t])} periods")
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: {e}")
    if not new_rows:
        print("No rows fetched.")
        return
    new_df = pd.DataFrame(new_rows)
    new_df["as_of_date"] = new_df["as_of_date"].map(_as_date)
    existing = load()
    existing["as_of_date"] = existing["as_of_date"].map(_as_date)

    # STRICTLY ADDITIVE: never drop an existing row. For overlapping
    # (ticker, as_of_date) only fill columns that are currently NaN.
    # Brand-new keys are appended. EDGAR/manual cells stay untouched.
    FILL_COLS = [
        "market_cap", "market_cap_b", "total_assets", "total_assets_b",
        "pb_ratio", "mktcap_to_assets", "ev_ebitda", "roe", "roic",
        "debt_to_equity", "shares_outstanding", "interest_coverage",
        "earnings_stability",
        "revenue_quarterly", "free_cash_flow", "capital_expenditure_ttm",
        "fcf_margin", "reinvestment_rate",
    ]
    idx = ["ticker", "as_of_date"]
    ex = existing.set_index(idx)
    nd = new_df.set_index(idx)
    overlap = ex.index.intersection(nd.index)
    n_filled = 0
    if len(overlap):
        src = nd.loc[overlap]
        for c in FILL_COLS:
            if c not in ex.columns:
                continue
            if c not in src.columns:
                continue
            missing = ex.loc[overlap, c].isna()
            if missing.any():
                take = src.loc[overlap, c].where(missing)
                n_here = int(take.notna().sum())
                if n_here:
                    ex.loc[overlap, c] = ex.loc[overlap, c].fillna(take)
                    n_filled += n_here
        existing = ex.reset_index()
    brand_new = new_df[~new_df.set_index(idx).index.isin(ex.index)].copy()
    if len(brand_new):
        combined = pd.concat([existing, brand_new], ignore_index=True)
    else:
        combined = existing
    # Ensure new columns exist in combined
    for c in new_df.columns:
        if c not in combined.columns:
            combined[c] = np.nan
    before = len(load())
    save(combined)
    after = len(load())
    print(f"Additive history: +{after - before} new rows, "
          f"{n_filled} NaN cells filled, "
          f"{len(new_df)} fetched for {new_df['ticker'].nunique()} tickers")


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
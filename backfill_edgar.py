#!/usr/bin/env python3
"""
backfill_edgar.py — Real decades-long point-in-time fundamentals from SEC EDGAR.

Pulls XBRL companyfacts per monitored ticker (via SEC's public JSON API,
no key — UA header required, 10 req/s limit) and computes as-of-quarter-end:

  ROE  = TTM NetIncomeLoss / StockholdersEquity
  ROIC = TTM NOPAT / InvestedCapital (NOPAT = OperatingIncomeLoss * 0.75)
  D/E  = TotalDebt / StockholdersEquity
  EV/EBITDA = (mktcap + debt - cash) / TTM EBITDA (EBITDA = OI + D&A)
  P/B  = mktcap / StockholdersEquity
  MktCap/Assets = mktcap / Assets
  interest_coverage = TTM OperatingIncomeLoss / TTM InterestExpenseNonOperating

Market cap = adj_close price × shares at the quarter-end (from
daily_prices.parquet; shares from EDGAR CommonStockSharesOutstanding /
OrdinarySharesNumber, fallback to current shares).

Rows are source=edgar and displace BOTH synthetic (fundamentals_history_backfill)
and shallow yfinance_history rows for the same (ticker, as_of_date) — EDGAR
is deeper (XBRL mandate ~2009+) and point-in-time, so it wins.

Usage:
  python backfill_edgar.py --max-tickers 50
  python backfill_edgar.py --tickers AAPL,MSFT
  python backfill_edgar.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
UA = {"User-Agent": "personal-research derek.moore@example.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Overrides: SEC's ticker->CIK map is sometimes stale/wrong. Known cases where
# the map points to a shell/related entity with thin XBRL history while the
# operating company's CIK holds the full history:
#   XOM -> 2115436 is 'ExxonMobil Holdings Corp' (shell); real opco is 34088.
#   AEP -> missing from map; real opco CIK 4904.
CIK_OVERRIDES = {
    "XOM": "0000034088",
    "AEP": "0000004904",
}

# income tags we can use for TTM (in priority order)
NI_TAGS = ["NetIncomeLoss"]
OI_TAGS = ["OperatingIncomeLoss", "OperatingIncome"]
DA_TAGS = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]
INT_TAGS = ["InterestExpenseNonOperating", "InterestExpense", "InterestAndDebtExpense"]
TAX_TAGS = ["IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitCurrentFederal", "IncomeTaxesPaid"]
PRETAX_TAGS = ["PretaxIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]
EQ_TAGS = ["StockholdersEquity", "CommonStockholdersEquity"]
ASSET_TAGS = ["Assets"]
DEBT_TAGS = ["LongTermDebtAndCapitalLeaseObligations", "LongTermDebt", "Debt"]
CASH_TAGS = ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"]
SHARES_TAGS = ["CommonStockSharesOutstanding", "OrdinarySharesNumber", "EntityCommonStockSharesOutstanding"]
INVESTED_TAGS = ["InvestedCapital"]


def load_cik_map() -> dict[str, str]:
    r = requests.get(TICKERS_URL, headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"]).zfill(10)
        out[str(row["ticker"]).upper()] = cik
    return out


def _facts(d: dict) -> dict:
    return d.get("facts", {}).get("us-gaap", {})


def _quarterly(tag_data: dict) -> pd.DataFrame:
    """Single-quarter rows from a us-gaap tag's USD entries.

    Income rows: keep only entries with a CYyyyyQn frame (single quarter);
    dedupe to one value per quarter (last filed wins). Balance rows: keep
    'end' as the as-of date.
    """
    rows = []
    for e in tag_data.get("units", {}).get("USD", []):
        f = e.get("frame", "")
        if "CY" in f and "Q" in f:  # single-quarter income frame
            q = f.split("CY")[1]  # e.g. 2026Q1
            rows.append({"end": e["end"], "frame": q, "val": e["val"], "filed": e.get("filed", "")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    return df.set_index("end")["val"]


def _balance_series(tag_data: dict) -> pd.Series:
    rows = []
    # shares are reported in 'shares' units, everything else in USD
    units = tag_data.get("units", {})
    for unit in ("USD", "shares"):
        for e in units.get(unit, []):
            rows.append({"end": e["end"], "val": e["val"], "filed": e.get("filed", "")})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    return df.set_index("end")["val"]


def fetch_ticker(ticker: str, cik: str) -> list[dict]:
    url = FACTS_URL.format(cik=cik)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    d = r.json()
    facts = _facts(d)

    def first_tag(tags: list[str]):
        for t in tags:
            if t in facts:
                return facts[t]
        return None

    def q(tags: list[str]) -> pd.DataFrame:
        td = first_tag(tags)
        return _quarterly(td) if td is not None else pd.DataFrame()

    def bal(tags: list[str]) -> pd.Series:
        td = first_tag(tags)
        return _balance_series(td) if td is not None else pd.Series(dtype=float)

    ni = q(NI_TAGS)
    oi = q(OI_TAGS)
    da = q(DA_TAGS)
    intexp = q(INT_TAGS)
    tax = q(TAX_TAGS)
    pretax = q(PRETAX_TAGS)
    eq = bal(EQ_TAGS)
    assets = bal(ASSET_TAGS)
    debt = bal(DEBT_TAGS)
    cash = bal(CASH_TAGS)
    shares = bal(SHARES_TAGS)
    invested = bal(INVESTED_TAGS)

    if ni.empty or eq.empty:
        return []

    # TTM income: sum of up to 4 consecutive quarters ending at each balance date
    ni_ttm = ni.rolling(4, min_periods=1).sum()
    oi_ttm = oi.rolling(4, min_periods=1).sum() if not oi.empty else pd.Series(dtype=float)
    da_ttm = da.rolling(4, min_periods=1).sum() if not da.empty else pd.Series(dtype=float)
    int_ttm = intexp.rolling(4, min_periods=1).sum() if not intexp.empty else pd.Series(dtype=float)
    tax_ttm = tax.rolling(4, min_periods=1).sum() if not tax.empty else pd.Series(dtype=float)
    pretax_ttm = pretax.rolling(4, min_periods=1).sum() if not pretax.empty else pd.Series(dtype=float)

    def ttm_at(series: pd.Series, qend) -> float | None:
        if series is None or series.empty:
            return None
        s = series[series.index <= qend].dropna()
        return float(s.iloc[-1]) if len(s) else None

    rows = []
    for qend, equity in eq.items():
        ttm_ni = ni_ttm.reindex(ni_ttm.index[ni_ttm.index <= qend]).dropna()
        ttm_ni = float(ttm_ni.iloc[-1]) if len(ttm_ni) else None
        ttm_oi = None
        if not oi_ttm.empty:
            s = oi_ttm[oi_ttm.index <= qend].dropna()
            ttm_oi = float(s.iloc[-1]) if len(s) else None
        ttm_da = None
        if not da_ttm.empty:
            s = da_ttm[da_ttm.index <= qend].dropna()
            ttm_da = float(s.iloc[-1]) if len(s) else None
        ttm_int = None
        if not int_ttm.empty:
            s = int_ttm[int_ttm.index <= qend].dropna()
            ttm_int = float(s.iloc[-1]) if len(s) else None

        def bal_at(series: pd.Series):
            if series is None or series.empty:
                return None
            s = series[series.index <= qend].dropna()
            return float(s.iloc[-1]) if len(s) else None

        e = bal_at(eq)
        a = bal_at(assets)
        db = bal_at(debt)
        ca = bal_at(cash)
        sh = bal_at(shares)
        inv = bal_at(invested)
        if e is None or e <= 0:
            continue
        # per-period effective tax rate: TTM tax / TTM pretax, clamped [0, 0.5];
        # fall back to 25% proxy when either side is unavailable.
        ttax = ttm_at(tax_ttm, qend)
        tpre = ttm_at(pretax_ttm, qend)
        eff_rate = None
        if ttax is not None and tpre and abs(tpre) > 1e-9:
            eff_rate = float(np.clip(ttax / tpre, 0.0, 0.5))
        rows.append({
            "qend": qend,
            "equity": e,
            "assets": a,
            "debt": db,
            "cash": ca,
            "shares": sh,
            "invested": inv,
            "ttm_ni": ttm_ni,
            "ttm_oi": ttm_oi,
            "ttm_da": ttm_da,
            "ttm_int": ttm_int,
            "eff_tax_rate": eff_rate,
        })
    return rows


def build_rows(ticker: str, frames: list[dict], px: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for fr in frames:
        qend = pd.Timestamp(fr["qend"])
        equity, assets, debt, cash = fr["equity"], fr["assets"], fr["debt"], fr["cash"]
        shares = fr["shares"]
        ttm_ni, ttm_oi, ttm_da = fr["ttm_ni"], fr["ttm_oi"], fr["ttm_da"]
        # market cap = price at qend × shares
        mcap = None
        if ticker in px and shares:
            p = px[ticker]
            avail = p[p.index <= qend]
            if len(avail):
                mcap = float(avail.iloc[-1]) * shares
        mcap_b = mcap / 1e9 if mcap else None
        roe = ttm_ni / equity if ttm_ni and equity else None
        rate = fr.get("eff_tax_rate") if fr.get("eff_tax_rate") is not None else 0.25
        if ttm_oi:
            nopat = ttm_oi * (1 - rate)
        elif ttm_ni is not None and fr.get("ttm_int"):
            # banks/energy often omit OperatingIncome; use NI + interest(1-t)
            nopat = ttm_ni + fr["ttm_int"] * (1 - rate)
        else:
            nopat = None
        invested = fr["invested"] if fr["invested"] else ((debt or 0) + equity)
        roic = nopat / invested if nopat and invested else None
        de = debt / equity if debt and equity else None
        ebitda = (ttm_oi + ttm_da) if ttm_oi is not None and ttm_da is not None else (ttm_oi if ttm_oi is not None else None)
        ev = (mcap + debt - cash) if mcap and debt is not None and cash is not None else None
        ev_ebitda = ev / ebitda if ev and ebitda else None
        pb = mcap / equity if mcap and equity else None
        mca = mcap / assets if mcap and assets else None
        icov = ttm_oi / fr["ttm_int"] if ttm_oi and fr["ttm_int"] else None
        rows.append({
            "ticker": ticker,
            "as_of_date": qend.date(),
            "market_cap": int(mcap) if mcap else None,
            "market_cap_b": round(mcap_b, 2) if mcap_b else None,
            "total_assets": int(assets) if assets else None,
            "total_assets_b": round(assets / 1e9, 2) if assets else None,
            "pb_ratio": round(pb, 3) if pb else None,
            "mktcap_to_assets": round(mca, 3) if mca else None,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
            "roe": round(roe, 4) if roe else None,
            "roic": round(roic, 4) if roic else None,
            "debt_to_equity": round(de, 3) if de else None,
            "interest_coverage": round(icov, 3) if icov else None,
            "source": "edgar",
            "notes": "SEC XBRL companyfacts (TTM income)",
            "last_updated": pd.Timestamp.now(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="Comma-separated subset")
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Fetch CIKs and report coverage, write nothing")
    args = ap.parse_args()

    stocks = pd.read_parquet(STOCKS_FILE) if STOCKS_FILE.exists() else pd.DataFrame()
    if stocks.empty or "ticker" not in stocks.columns:
        raise SystemExit("monitored_stocks.parquet missing")
    tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.max_tickers:
        tickers = tickers[: args.max_tickers]

    print("Loading SEC ticker→CIK map...")
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    matched = [(t, cik_map[t]) for t in tickers if t in cik_map]
    unmatched = [t for t in tickers if t not in cik_map]
    print(f"  {len(matched)}/{len(tickers)} tickers have a CIK")
    if unmatched:
        print(f"  no CIK ({len(unmatched)}): {', '.join(unmatched)}")
        print("    (ETFs/funds and delisted ADRs have no XBRL statements; expected gaps)")
    if args.dry_run:
        return

    # price series for mktcap
    from analytics_common import load_adj_prices_pandas
    prices = load_adj_prices_pandas(tickers=tickers)
    px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}

    all_rows: list[dict] = []
    ok = 0
    for t, cik in matched:
        try:
            frames = fetch_ticker(t, cik)
            rows = build_rows(t, frames, px)
            if rows:
                all_rows.extend(rows)
                ok += 1
                print(f"  {t}: {len(rows)} quarters (cik={cik})")
            time.sleep(0.12)  # SEC: ≤10 req/s
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: {e}")
            time.sleep(0.12)
    if not all_rows:
        print("No rows fetched.")
        return

    new_df = pd.DataFrame(all_rows)
    existing = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    real_tickers = set(new_df["ticker"])
    # drop synthetic backfill AND shallow yfinance_history for these tickers
    existing = existing[
        ~(
            existing["ticker"].isin(real_tickers)
            & (existing.get("source").isin(["fundamentals_history_backfill", "yfinance_history"]))
        )
    ]
    keys = set(zip(new_df["ticker"], new_df["as_of_date"]))
    existing = existing[~existing.apply(lambda r: (r["ticker"], r["as_of_date"]) in keys, axis=1)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    # DATE-native as_of_date; normalize last_updated to Timestamp
    combined["as_of_date"] = pd.to_datetime(combined["as_of_date"]).dt.date
    if "last_updated" in combined.columns:
        combined["last_updated"] = pd.to_datetime(combined["last_updated"], errors="coerce")
    combined = combined.sort_values(["ticker", "as_of_date"]).drop_duplicates(
        subset=["ticker", "as_of_date"], keep="last"
    )
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), FUND)
    print(f"EDGAR backfill: {len(new_df)} rows for {len(real_tickers)} tickers → {FUND}")
    print(f"  total fundamentals rows now: {len(combined)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""update_polygon_financials.py — Additive quarterly fundamentals from
Massive/Polygon `GET /vX/reference/financials`.

Why: yfinance history is ~5–7 quarters; EDGAR is deep but US-filer only.
Polygon financials fill missing (ticker, quarter) rows and NaN cells for
the full universe without overwriting EDGAR.

Auth (never printed, never committed):
  POLYGON_API_KEY, or massive_credentials.json secret_access_key
  (same value as the REST key on this account).

Usage:
  python update_polygon_financials.py
  python update_polygon_financials.py --tickers AAPL,GOLD --max-pages 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from update_fundamentals import _as_date, load, save, universe_tickers

DATA_DIR = Path(__file__).resolve().parent
CRED = DATA_DIR / "massive_credentials.json"
URL = "https://api.polygon.io/vX/reference/financials"
SLEEP = 12.5  # stay under the minute cap on this plan


def _api_key() -> str:
    k = os.environ.get("POLYGON_API_KEY", "").strip()
    if k:
        return k
    if CRED.exists():
        d = json.loads(CRED.read_text())
        k = (d.get("secret_access_key") or "").strip()
        if k:
            return k
    raise SystemExit("No Polygon REST key (POLYGON_API_KEY or massive_credentials.json)")


def _val(block: dict | None, *names) -> float | None:
    if not isinstance(block, dict):
        return None
    for n in names:
        item = block.get(n)
        if isinstance(item, dict) and item.get("value") is not None:
            try:
                return float(item["value"])
            except (TypeError, ValueError):
                continue
        if item is not None and not isinstance(item, dict):
            try:
                return float(item)
            except (TypeError, ValueError):
                continue
    return None


def _get(url: str, params: dict, key: str) -> dict:
    for attempt in range(6):
        r = requests.get(url, params=params, timeout=45)
        if r.status_code == 429:
            wait = SLEEP * (attempt + 1)
            print(f"  rate-limited, sleep {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            raise SystemExit("Polygon financials 401 — key not authorized for this endpoint")
        r.raise_for_status()
        return r.json()
    return {}


def fetch_ticker(ticker: str, key: str, max_pages: int) -> list[dict]:
    params = {
        "ticker": ticker,
        "timeframe": "quarterly",
        "limit": 100,
        "order": "asc",
        "sort": "filing_date",
        "apiKey": key,
    }
    out: list[dict] = []
    url, pages = URL, 0
    while url and pages < max_pages:
        body = _get(url, params if pages == 0 else {"apiKey": key}, key)
        pages += 1
        for rec in body.get("results") or []:
            fin = rec.get("financials") or {}
            inc = fin.get("income_statement") or {}
            bal = fin.get("balance_sheet") or {}
            qend = rec.get("end_date") or rec.get("filing_date")
            ni = _val(inc, "net_income_loss", "net_income_loss_attributable_to_parent")
            oi = _val(inc, "operating_income_loss", "operating_income_quarterly")
            ebitda = _val(inc, "ebitda")
            shares = _val(inc, "basic_average_shares", "diluted_average_shares")
            equity = _val(
                bal,
                "equity_attributable_to_parent",
                "shareholders_equity",
                "shareholders_equity",
            )
            assets = _val(bal, "total_assets", "liabilities_and_equity")
            debt = _val(bal, "long_term_debt", "noncurrent_liabilities")
            cash = _val(bal, "cash_and_equivalents", "cash_and_equivalents")
            if shares is None:
                shares = _val(bal, "outstanding_shares")
            # TTM-ish: we only have this quarter here; ratios use this quarter's
            # income * 4 as a last-resort only when TTM is impossible. Prefer
            # leaving income ratios None if we cannot form TTM from the batch.
            out.append({
                "ticker": ticker,
                "as_of_date": _as_date(qend),
                "ni_q": ni,
                "oi_q": oi,
                "ebitda_q": ebitda,
                "shareholders_equity": equity,
                "total_assets": assets,
                "debt": debt,
                "cash_and_equivalents": cash,
                "shares": shares,
                "source": "polygon_financials",
                "notes": "polygon vX quarterly financials",
                "last_updated": pd.Timestamp.now(),
            })
        nxt = body.get("next_url")
        url = nxt
        params = {"apiKey": key} if nxt else params
        if nxt:
            time.sleep(SLEEP)
    return out


def _ttm(rows: list[dict], field: str) -> list[float | None]:
    vals = [r.get(field) for r in rows]
    out = []
    for i, _ in enumerate(vals):
        window = [v for v in vals[max(0, i - 3): i + 1] if v is not None]
        out.append(sum(window) if len(window) >= 2 else (window[0] if window else None))
    return out


def finalize(rows: list[dict], prices: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    rows = sorted(rows, key=lambda r: (r["ticker"], r["as_of_date"] or pd.Timestamp(0).date()))
    by_t: dict[str, list] = {}
    for r in rows:
        by_t.setdefault(r["ticker"], []).append(r)
    out = []
    for t, grp in by_t.items():
        ni = _ttm(grp, "ni_q")
        oi = _ttm(grp, "oi_q")
        eb = _ttm(grp, "ebitda_q")
        for r, ttm_ni, ttm_oi, ttm_eb in zip(grp, ni, oi, eb):
            equity, assets, debt, cash, shares = (
                r.get("shareholders_equity"), r.get("total_assets"), r.get("debt"),
                r.get("cash_and_equivalents"), r.get("shares"),
            )
            mcap = None
            qend = r.get("as_of_date")
            if t in prices and shares and qend:
                p = prices[t]
                avail = p[p.index <= pd.Timestamp(qend)]
                if len(avail):
                    mcap = float(avail.iloc[-1]) * shares
            nopat = ttm_oi * 0.75 if ttm_oi is not None else None
            roe = ttm_ni / equity if ttm_ni and equity else None
            roic = nopat / (equity + (debt or 0) - (cash or 0)) if nopat and equity else None
            de = debt / equity if debt and equity else None
            ev = (mcap + (debt or 0) - (cash or 0)) if mcap else None
            ev_ebitda = ev / ttm_eb if ev and ttm_eb else None
            pb = mcap / equity if mcap and equity else None
            mca = mcap / assets if mcap and assets else None
            out.append({
                "ticker": t,
                "as_of_date": qend,
                "market_cap": int(mcap) if mcap else None,
                "market_cap_b": round(mcap / 1e9, 2) if mcap else None,
                "total_assets": int(assets) if assets else None,
                "total_assets_b": round(assets / 1e9, 2) if assets else None,
                "pb_ratio": round(pb, 3) if pb else None,
                "mktcap_to_assets": round(mca, 3) if mca else None,
                "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
                "roe": round(roe, 4) if roe else None,
                "roic": round(roic, 4) if roic else None,
                "debt_to_equity": round(de, 3) if de else None,
                # guard the same corruption class that made the old `shares`
                # column unusable (values up to 7.96e14); no US listed company
                # has >=1e11 shares outstanding.
                "shares_outstanding": shares if (shares and 0 < shares < 1e11) else None,
                "source": "polygon_financials",
                "notes": r.get("notes"),
                "last_updated": r.get("last_updated"),
            })
    return pd.DataFrame(out)


def merge_additive(existing: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if new_df.empty:
        return existing, 0, 0
    new_df = new_df.copy()
    new_df["as_of_date"] = new_df["as_of_date"].map(_as_date)
    existing = existing.copy()
    existing["as_of_date"] = existing["as_of_date"].map(_as_date)
    FILL = [
        "market_cap", "market_cap_b", "total_assets", "total_assets_b",
        "pb_ratio", "mktcap_to_assets", "ev_ebitda", "roe", "roic",
        "debt_to_equity", "shares_outstanding", "interest_coverage",
        "earnings_stability",
    ]
    idx = ["ticker", "as_of_date"]
    ex = existing.set_index(idx)
    nd = new_df.set_index(idx)
    overlap = ex.index.intersection(nd.index)
    n_filled = 0
    if len(overlap):
        src = nd.loc[overlap]
        for c in FILL:
            if c not in ex.columns or c not in src.columns:
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
    n_new = len(brand_new)
    combined = pd.concat([existing, brand_new], ignore_index=True) if n_new else existing
    return combined, n_new, n_filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers")
    ap.add_argument("--max-tickers", type=int)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--missing-only", action="store_true",
                    help="Only tickers with no fundamentals row")
    args = ap.parse_args()
    key = _api_key()
    tickers = universe_tickers()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.missing_only:
        have = set(load()["ticker"].astype(str).str.upper())
        tickers = [t for t in tickers if t not in have]
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"polygon financials: {len(tickers)} tickers")

    try:
        from analytics_common import load_adj_prices_pandas
        prices = load_adj_prices_pandas(tickers=tickers)
        px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}
    except Exception as e:
        print(f"  price load failed: {e}")
        px = {}

    raw: list[dict] = []
    for i, t in enumerate(tickers, 1):
        try:
            rows = fetch_ticker(t, key, args.max_pages)
            print(f"  [{i}/{len(tickers)}] {t}: {len(rows)} quarters")
            raw.extend(rows)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t}: {e}")
        time.sleep(SLEEP)

    new_df = finalize(raw, px)
    existing = load()
    before = len(existing)
    combined, n_new, n_filled = merge_additive(existing, new_df)
    save(combined)
    after = len(load())
    print(f"Additive polygon financials: +{after - before} rows "
          f"(brand-new keys {n_new}), {n_filled} NaN cells filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

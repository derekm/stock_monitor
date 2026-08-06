#!/usr/bin/env python3
"""
filings_sentiment.py — Lightweight lexicon sentiment on recent SEC 8-K
filings for the monitored universe.

Why it exists: the architecture TODO "alternative data — NLP on filings".
Full NLP is heavy; this is the honest MVP: pull the latest 8-K (current
report) for each ticker from SEC EDGAR full-text search, extract the plain
text, and score it with the classic Loughran-McDonald financial word lists
(negative/positive) + a small forward-looking list. Output is a per-ticker
sentiment snapshot.

Notes on honesty:
  * 8-Ks are usually boilerplate (appointments, notices); sentiment is often
    near-neutral — that IS the signal (no news beats bad news).
  * Lexicon matching is case-insensitive word-boundary; no lemmatization.

Output: filings_sentiment.csv — (ticker, filing_date, form, n_neg, n_pos,
  n_fwd, score_neg_minus_pos_per_1k)

Usage:
    python filings_sentiment.py [--save] [--tickers AAPL,MSFT] [--limit 5]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

from analytics_common import DATA_DIR

UA = {"User-Agent": "personal-research derek.moore@example.com"}
SEARCH = "https://efts.sec.gov/LATEST/search-index"
OUT = DATA_DIR / "filings_sentiment.csv"

# Loughran-McDonald-style word lists (subset, financial-domain tuned)
NEG = {"loss", "losses", "impairment", "impairments", "restructuring", "litigation",
       "default", "bankruptcy", "restated", "restatement", "weakness", "weaknesses",
       "deterioration", "decline", "declined", "risk", "risks", "materially",
       "adversely", "termination", "delisted", "delisting", "fraud", "misstatement",
       "charge", "charges", "write-down", "writedown", "investigation", "penalty",
       "penalties", "violation", "violations", "noncompliance", "liquidation",
       "going concern", "going-concern", "recession", "layoffs", "downgrade"}
POS = {"growth", "growths", "record", "records", "strong", "strengthen", "improved",
       "improvement", "improvements", "increase", "increased", "exceeded", "beat",
       "beats", "guidance", "raised", "raise", "expansion", "expanding", "launch",
       "launched", "win", "won", "awarded", "renewed", "partnership", "partnerships",
       "acquisition", "acquisitions", "synergy", "synergies", "momentum", "outperform"}
FWD = {"expects", "expected", "anticipates", "anticipate", "guidance", "outlook",
       "forecast", "projects", "projected", "plans", "plan", "intends", "intend",
       "will", "believes", "believe"}


def cik_for_ticker(ticker: str) -> str | None:
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
        for v in r.json().values():
            if str(v["ticker"]).upper() == ticker.upper():
                return str(v["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def recent_8k_accessions(cik: str, limit: int = 3) -> list[tuple[str, str]]:
    """(accession_no, filing_date) for the most recent 8-K filings."""
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA, timeout=30)
        recent = r.json().get("filings", {}).get("recent", {})
    except Exception:
        return []
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    out = []
    for i, f in enumerate(forms):
        if f == "8-K" and i < len(accs):
            out.append((str(accs[i]).replace("-", ""), str(dates[i])))
        if len(out) >= limit:
            break
    return out


def fetch_8k_text(cik: str, accession: str) -> str:
    """Fetch the primary 8-K document body text for an accession."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"
    try:
        idx = requests.get(f"{url}/index.json", headers=UA, timeout=30).json()
        items = idx.get("directory", {}).get("item", [])
        docs = [it["name"] for it in items
                if it.get("name", "").endswith((".htm", ".html"))]
        if not docs:
            return ""
        # the substantive body is the longest htm (exhibit press release);
        # concat all, scripts/styles stripped
        best = ""
        for doc in docs:
            try:
                txt = requests.get(f"{url}/{doc}", headers=UA, timeout=30).text
                txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", txt)
                txt = re.sub(r"<[^>]+>", " ", txt)
                if len(txt) > len(best):
                    best = txt
            except Exception:
                continue
        return best
    except Exception:
        return ""


def latest_8k_text(ticker: str, cik: str | None = None, limit: int = 3) -> list[dict]:
    if not cik:
        cik = cik_for_ticker(ticker)
    if not cik:
        return []
    out = []
    for acc, fdate in recent_8k_accessions(cik, limit=limit):
        text = fetch_8k_text(cik, acc)
        out.append({
            "ticker": ticker,
            "filing_date": fdate,
            "form": "8-K",
            "text": text,
        })
    return out


def score_text(text: str) -> dict:
    tokens = re.findall(r"[a-z][a-z-]+", text.lower())
    n_neg = sum(1 for t in tokens if t in NEG)
    n_pos = sum(1 for t in tokens if t in POS)
    n_fwd = sum(1 for t in tokens if t in FWD)
    n = max(len(tokens), 1)
    return {
        "n_neg": n_neg, "n_pos": n_pos, "n_fwd": n_fwd,
        "score_per_1k": round((n_neg - n_pos) / n * 1000, 3),
    }


def build(tickers: list[str], limit: int = 3) -> pd.DataFrame:
    rows = []
    for t in tickers:
        filings = latest_8k_text(t, limit=limit)
        for f in filings:
            s = score_text(f.pop("text"))
            f.update(s)
            rows.append(f)
        time.sleep(0.11)  # SEC rate limit
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = monitored universe")
    ap.add_argument("--limit", type=int, default=3, help="8-Ks per ticker")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        stocks = DATA_DIR / "monitored_stocks.parquet"
        tickers = sorted(pd.read_parquet(stocks)["ticker"].astype(str).str.upper().unique()) if stocks.exists() else []
    df = build(tickers, limit=args.limit)
    print(f"=== Filings sentiment ({len(df)} filings) ===")
    cols = [c for c in ["ticker", "filing_date", "form", "n_neg", "n_pos", "n_fwd", "score_per_1k"] if c in df]
    print(df[cols].head(20).to_string(index=False) if len(df) else "(no filings matched)")
    if args.save and len(df):
        df.to_csv(OUT, index=False)
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

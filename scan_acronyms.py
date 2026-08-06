#!/usr/bin/env python3
"""Scan docs + code for acronym candidates (all-caps tokens), report frequency."""
import re, collections, os

ROOTS = [
    r"C:/Users/derek/src/stockmagic",
]
SKIP_EXT = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".json", ".parquet", ".csv", ".svg", ".woff", ".woff2", ".ttf"}
SKIP_DIRS = {".git", ".venv", "node_modules", "dashboard_data", "research", "__pycache__", ".hermes"}

ACRO = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Z0-9]+)*\b")
# tokens that are almost always tickers / noise
TICKERISH = set("""
AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V MA WMT JNJ PG KO PEP XOM CVX UNH HD MCD DIS NKE BA
GE CAT MMM INTC AMD MU AVGO ORCL CRM ADBE NFLX QCOM TXN CSCO IBM AMAT LRCX KLAC ASML TSM
RF SMCI SPCX AEP DUK SO NEE D VZ T KHC CAG HMC MOS PFE BAYRY BRK ABBV MRK LLY GILD AMGN
SPY QQQ DIA IWM EFA EEM VNQ XLE XLF XLP XLU XLV XBI XLI XLY XLB XLK XLC XLU VIG VYM SCHD
DVY SPLV USMV VUG IYR ARKK VOO VTI BND GLD SLV TLT IEF LQD HYG JNK EEM IYR
AFL ALL AMAT AMD ANDE ASIX ASTS C CW DUK HD IVZ JPM KO MMM MSFT NVR O REGN RF STLD T
UNH VZ WM XOM FICO AEP
""".split())

def scan(path, prefix=""):
    counts = collections.Counter()
    files = 0
    for root, dirs, files_ in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files_:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, path)
            if fn.startswith("_"):
                continue
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    txt = f.read()
            except Exception:
                continue
            files += 1
            for m in ACRO.finditer(txt):
                tok = m.group(0).strip("-_/")
                if len(tok) < 2:
                    continue
                counts[tok] += 1
    return counts, files

if __name__ == "__main__":
    counts, files = scan(ROOTS[0])
    print(f"files scanned: {files}")
    print("== acronym candidates (freq >= 3, not tickerish) ==")
    for tok, n in counts.most_common():
        if n < 3 or tok in TICKERISH:
            continue
        # skip pure numbers like 2026
        if tok.isdigit():
            continue
        print(f"{n:5d}  {tok}")

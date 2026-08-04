"""
parse_sp500_changes.py — build the authoritative S&P 500 ADD/REMOVE event log.

PRIMARY SOURCE: tickerleague.com/indices/stock/sp-500/additions-and-removals
  - Official S&P index announcements, embedded as a JSON array in the page
    (client-paginated over 31 pages, back to 1957). 1,500+ real events.
  - The HTML table only shows 50 rows; the FULL dataset is in a <script> tag.

FALLBACK: the Wikipedia "List of S&P 500 companies" changes table (1976-2026),
  used only if tickerleague is unreachable.

Output: sp500_changes.parquet
  event_date DATE, added VARCHAR, removed VARCHAR, reason VARCHAR

We DO NOT fabricate removals; every row is a real announced change.
Ticker normalization: trim whitespace; cast NULL/null -> None (pure addition /
unknown reason). Downstream joins map to canonical sp500_constituents tickers.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "sp500_changes.parquet"


def _norm_ticker(t):
    if t is None:
        return None
    t = str(t).strip().upper()
    if t in ("", "NULL", "NONE", "NA"):
        return None
    return t


def _norm_reason(r):
    if r is None:
        return None
    r = str(r).strip()
    if r.lower() in ("null", "none", "na", ""):
        return None
    return r


def load_tickerleague() -> pd.DataFrame:
    """Fetch + parse the full history from tickerleague."""
    from parse_tickerleague_changes import parse_tickerleague

    df = parse_tickerleague()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "event_date": r["event_date"],
                "added": _norm_ticker(r.get("added_ticker")),
                "removed": _norm_ticker(r.get("removed_ticker")),
                "reason": _norm_reason(r.get("reason")),
            }
        )
    out = pd.DataFrame(rows)
    return out


def load_wikipedia() -> pd.DataFrame:
    """Fallback: parse the Wikipedia changes table from a previously saved html."""
    html = HERE / "sp500_main.html"
    if not html.exists():
        return pd.DataFrame(columns=["event_date", "added", "removed", "reason"])
    # (kept minimal; the old HTMLParser path from before)
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self._row = []
            self._in_td = False
            self._buf = ""
            self._in_wt = False
            self._in_target = False
            self.rows = []
            self._hdr = False

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                cls = dict(attrs).get("class", "")
                if "wikitable" in cls:
                    self._in_wt = True
            if self._in_wt and tag in ("td", "th"):
                self._in_td = True
                self._buf = ""

        def handle_endtag(self, tag):
            if self._in_wt and tag in ("td", "th"):
                self._row.append(self._buf.strip())
                self._in_td = False
            if tag == "tr" and self._in_wt:
                if self._row:
                    self.rows.append(self._row)
                self._row = []
            if tag == "table" and self._in_wt:
                self._in_wt = False

        def handle_data(self, data):
            if self._in_td:
                self._buf += data

    p = _P()
    p.feed(open(html, encoding="utf-8", errors="ignore").read())
    # naive: find rows with a date + added/removed cells (best-effort)
    out = []
    for r in p.rows:
        if len(r) < 4:
            continue
        date = r[0]
        try:
            dt.datetime.strptime(date, "%B %d, %Y")
        except Exception:
            continue
        out.append(
            {
                "event_date": dt.datetime.strptime(date, "%B %d, %Y").date().isoformat(),
                "added": _norm_ticker(r[1]) if len(r) > 1 else None,
                "removed": _norm_ticker(r[2]) if len(r) > 2 else None,
                "reason": _norm_reason(r[3]) if len(r) > 3 else None,
            }
        )
    return pd.DataFrame(out)


def main():
    df = load_tickerleague()
    if df.empty:
        print("tickerleague empty; falling back to Wikipedia")
        df = load_wikipedia()
    # dedupe by (event_date, added, removed)
    df = df.dropna(subset=["event_date"]).drop_duplicates(
        subset=["event_date", "added", "removed"]
    )
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    df = df.sort_values("event_date").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows, {df.event_date.min()} -> {df.event_date.max()})")
    print(f"  adds={df.added.notna().sum()} removes={df.removed.notna().sum()}")


if __name__ == "__main__":
    main()

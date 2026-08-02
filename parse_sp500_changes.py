"""
parse_sp500_changes.py — parse the S&P 500 historical ADD/REMOVE (changes)
table from the Wikipedia "List of S&P 500 companies" page.

This is the real inclusion/exclusion event log (Effective Date, Added,
Removed, Reason) that the membership-timeseries simulation needs. We only had
additions (via date_added) before; this fills the REMOVALS gap.

Output: sp500_changes.parquet
  event_date DATE, added VARCHAR, removed VARCHAR, reason VARCHAR

Ticker normalization: Wikipedia occasionally lists spinoff/segment tickers
(e.g. HONA for the Honeywell Aerospace spin). We keep tickers as-is but strip
whitespace; downstream joins should map to our canonical sp500_constituents
tickers. We do NOT fabricate removals.
"""
from __future__ import annotations
import datetime as dt
import re
from html.parser import HTMLParser
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
HTML = HERE / "sp500_main.html"
OUT = HERE / "sp500_changes.parquet"


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)        # strip tags
    s = re.sub(r"\[\d+\]", "", s)        # strip [1] refs
    s = s.replace("\xa0", " ")
    return s.strip()


class ChangesParser(HTMLParser):
    """Grab the wikitable whose header contains 'Effective Date' + 'Added' + 'Removed'."""

    def __init__(self):
        super().__init__()
        self._in_wt = False
        self._in_target = False
        self._in_row = False
        self._in_cell = False
        self._cell = []
        self._row = []
        self._header_cells = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            cls = dict(attrs).get("class", "")
            if "wikitable" in cls:
                self._in_wt = True
        if not self._in_wt:
            return
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if not self._in_wt:
            return
        if tag in ("td", "th"):
            txt = _clean("".join(self._cell))
            self._row.append(txt)
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self._emit()
            self._in_row = False
        elif tag == "table":
            self._in_wt = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)

    def _emit(self):
        if not self._row:
            return
        low = [c.lower() for c in self._row]
        # detect the two-row header. Row 1 cells include 'effective date' and 'reason'.
        if not self._header_cells:
            if any("effective date" in c for c in low) and any("reason" in c for c in low):
                self._header_cells = low
                self._in_target = True
                return
            # second header row (Added Ticker/Security, Removed Ticker/Security)
            if self._in_target and ("ticker" in low or "security" in low):
                return
        if not self._in_target or not self._header_cells:
            return
        # data row: [Effective Date, AddedTicker, AddedSecurity, RemovedTicker, RemovedSecurity, Reason]
        if len(self._row) < 6:
            return
        date_s, a_tk, a_sec, r_tk, r_sec, reason = self._row[:6]
        d = _parse_date(date_s)
        if d is None:
            return
        added = a_tk.strip() or None
        removed = r_tk.strip() or None
        if added is None and removed is None:
            return
        self.rows.append({
            "event_date": d,
            "added": added,
            "removed": removed,
            "reason": reason.strip() or None,
        })


def _parse_date(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return dt.date.fromisoformat(dt.datetime.strptime(s, fmt).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return None


def main():
    html = HTML.read_text(encoding="utf-8", errors="ignore")
    p = ChangesParser()
    p.feed(html)
    print(f"parsed {len(p.rows)} change events")
    con = duckdb.connect()
    con.execute("CREATE TABLE chg (event_date DATE, added VARCHAR, removed VARCHAR, reason VARCHAR)")
    con.executemany(
        "INSERT INTO chg VALUES (?, ?, ?, ?)",
        [(r["event_date"], r["added"], r["removed"], r["reason"]) for r in p.rows],
    )
    con.execute(f"COPY (SELECT * FROM chg ORDER BY event_date) TO '{OUT.as_posix()}' (FORMAT PARQUET)")
    print(f"wrote {OUT} ({len(p.rows)} rows)")
    print("date range:", con.execute("SELECT MIN(event_date), MAX(event_date) FROM chg").fetchone())
    print("adds:", con.execute("SELECT COUNT(*) FROM chg WHERE added IS NOT NULL").fetchone()[0],
          "removes:", con.execute("SELECT COUNT(*) FROM chg WHERE removed IS NOT NULL").fetchone()[0])
    print("sample (recent):", con.execute("SELECT * FROM chg ORDER BY event_date DESC LIMIT 3").fetchall())


if __name__ == "__main__":
    main()

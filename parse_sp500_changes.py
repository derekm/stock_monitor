"""Parse the 'Selected changes to the list of S&P 500 components' table from
the Wikipedia S&P 500 page. That table is the real historical inclusion/
exclusion EVENT log (date, added ticker, removed ticker, reason) — the missing
timeseries our simulation needs (we only had additions via date_added).

Outputs sp500_changes.parquet:
  event_date (DATE), action (always 'change'), added (VARCHAR or NULL),
  removed (VARCHAR or NULL), reason (VARCHAR)
"""
from __future__ import annotations
import datetime as dt
import re
from html.parser import HTMLParser
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
HTML = HERE / "sp500_page.html"
OUT = HERE / "sp500_changes.parquet"


class ChangesParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.grabbing = False
        self.in_row = False
        self.in_cell = False
        self.cell = []
        self.row = []
        self.header = None
        self.rows = []
        self._tbl_count = 0
        self._in_wikitable = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            cls = dict(attrs).get("class", "")
            if "wikitable" in cls:
                self._in_wikitable = True
                self._tbl_count += 1
        if not self._in_wikitable:
            return
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.cell = []

    def handle_endtag(self, tag):
        if not self._in_wikitable:
            return
        if tag in ("td", "th"):
            txt = self._clean("".join(self.cell))
            self.row.append(txt)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self._emit(self.row)
            self.in_row = False
        elif tag == "table":
            self._in_wikitable = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def _emit(self, cells):
        if not cells:
            return
        low = [c.strip().lower() for c in cells]
        is_changes_header = ("date" in low) and ("added" in low) and ("removed" in low)
        if self.header is None:
            if is_changes_header:
                self.header = low
                self.grabbing = True
            return
        if not self.grabbing:
            return
        rec = {self.header[i]: (cells[i] if i < len(cells) else "") for i in range(len(self.header))}
        d = parse_date(rec.get("date"))
        added = (rec.get("added") or "").strip() or None
        removed = (rec.get("removed") or "").strip() or None
        if d is None and not (added or removed):
            return
        self.rows.append({
            "event_date": d,
            "added": added,
            "removed": removed,
            "reason": (rec.get("reason") or "").strip() or None,
        })

    @staticmethod
    def _clean(s):
        s = re.sub(r"\[\d+\]", "", s)
        s = s.replace("\xa0", " ")
        return s.strip()


def parse_date(s):
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
    recs = [r for r in p.rows if r["event_date"]]
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE chg (event_date DATE, added VARCHAR, removed VARCHAR, reason VARCHAR)"
    )
    con.executemany(
        "INSERT INTO chg VALUES (?, ?, ?, ?)",
        [(r["event_date"], r["added"], r["removed"], r["reason"]) for r in recs],
    )
    con.execute(f"COPY (SELECT * FROM chg ORDER BY event_date) TO '{OUT.as_posix()}' (FORMAT PARQUET)")
    print(f"wrote {OUT} ({len(recs)} rows)")
    print("date range:", con.execute("SELECT MIN(event_date), MAX(event_date) FROM chg").fetchone())
    print("sample:", con.execute("SELECT * FROM chg ORDER BY event_date DESC LIMIT 3").fetchall())


if __name__ == "__main__":
    main()

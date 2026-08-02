"""Parse the S&P 500 constituents table from the downloaded Wikipedia HTML.

Robust to the inconsistent column layout (some rows carry an extra
'Headquarters' column). Strategy: locate the first wikitable, read its header
row by matching known header texts, then map every data row's cells by POSITION
against that header. Unknown/ignored headers (Headquarters) are skipped.
"""
from __future__ import annotations
import datetime as dt
import re
from html.parser import HTMLParser
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
HTML = HERE / "sp500_wiki.html"
OUT = HERE / "sp500_constituents.parquet"

HEADER_MAP = {
    "symbol": "ticker",
    "security": "name",
    "gics sector": "gics_sector",
    "gics sub-industry": "gics_sub_industry",
    "date added": "date_added",
    "cik": "cik",
    "founded": "founded",
}


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.table_done = False
        self.in_row = False
        self.in_cell = False
        self.cell = []
        self.row_cells = []          # raw texts for current row
        self.header = None           # list of semantic names aligned to positions
        self.rows = []               # list of dicts
        self._seen_first_table = False

    def handle_starttag(self, tag, attrs):
        if tag == "table" and not self.in_table and not self._seen_first_table:
            cls = dict(attrs).get("class", "")
            if "wikitable" in cls:
                self.in_table = True
        if not self.in_table or self.table_done:
            return
        if tag == "tr":
            self.in_row = True
            self.row_cells = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.cell = []

    def handle_endtag(self, tag):
        if not self.in_table or self.table_done:
            return
        if tag in ("td", "th"):
            txt = self._clean("".join(self.cell))
            self.row_cells.append(txt)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self._emit_row(self.row_cells)
            self.in_row = False
        elif tag == "table":
            self.in_table = False
            self.table_done = True   # only the first wikitable

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def _emit_row(self, cells):
        if not cells:
            return
        # Identify header row: at least 2 known header texts present.
        lowered = [c.strip().lower() for c in cells]
        known = sum(1 for c in lowered if c in HEADER_MAP)
        if self.header is None and known >= 2:
            self.header = [HEADER_MAP.get(c, None) for c in lowered]
            return
        if self.header is None:
            return  # skip pre-header rows
        # data row: map by position
        rec = {}
        for idx, name in enumerate(self.header):
            if name is None:
                continue
            rec[name] = cells[idx] if idx < len(cells) else None
        if rec.get("ticker"):
            self.rows.append(rec)

    @staticmethod
    def _clean(s):
        s = re.sub(r"\[\d+\]", "", s)
        s = s.replace("\xa0", " ")
        return s.strip()


def parse_date(s):
    s = (s or "").strip()
    if not s or s in ("—", "-"):
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.date.fromisoformat(dt.datetime.strptime(s, fmt).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return None


def main():
    html = HTML.read_text(encoding="utf-8", errors="ignore")
    p = P()
    p.feed(html)
    print(f"parsed {len(p.rows)} data rows")

    records = []
    for d in p.rows:
        rec = {k: d.get(k) for k in HEADER_MAP.values()}
        rec["date_added"] = parse_date(rec.get("date_added"))
        rec["founded"] = (rec.get("founded") or "").strip() or None
        rec["current"] = True
        rec["ticker"] = (rec.get("ticker") or "").upper().strip()
        if not rec["name"] or "<" in rec["name"]:
            continue
        records.append(rec)

    tset = [x["ticker"] for x in records]
    assert len(tset) == len(set(tset)), "duplicate tickers!"
    assert 490 <= len(records) <= 505, f"unexpected count {len(records)}"

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE constituents (
            ticker VARCHAR, name VARCHAR, gics_sector VARCHAR,
            gics_sub_industry VARCHAR, date_added DATE, cik VARCHAR,
            founded VARCHAR, current BOOLEAN
        )
        """
    )
    con.executemany(
        "INSERT INTO constituents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (r["ticker"], r["name"], r["gics_sector"], r["gics_sub_industry"],
             r["date_added"], r["cik"], r["founded"], r["current"])
            for r in records
        ],
    )
    con.execute(f"COPY (SELECT * FROM constituents ORDER BY ticker) TO '{OUT.as_posix()}' (FORMAT PARQUET)")
    print(f"wrote {OUT} ({len(records)} rows)")
    print("sectors:", con.execute("SELECT gics_sector, COUNT(*) c FROM constituents GROUP BY 1 ORDER BY c DESC LIMIT 5").fetchall())
    print("date_added null:", con.execute("SELECT COUNT(*) FROM constituents WHERE date_added IS NULL").fetchone()[0])


if __name__ == "__main__":
    main()

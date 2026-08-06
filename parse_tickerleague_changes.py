#!/usr/bin/env python3
"""Extract the FULL S&P 500 additions & removals history from tickerleague.com
(the data is embedded as a JS-stringified JSON array in a <script> tag; the site
paginates client-side over 31 pages back to the 1950s).

Robust parse: strict json.loads fails on a handful of rows that contain an
inner quoted phrase (e.g. 'changed its name to the "new" Ingersoll Rand').
We use a tolerant field scanner instead of a strict parser.
"""
import requests, re, json, sys, datetime as dt
import pandas as pd

BASE = "https://tickerleague.com/indices/stock/sp-500/additions-and-removals"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def extract_array(html: str):
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    big = max((s for s in scripts if "date_added" in s), key=len)
    start = big.rfind("[", 0, big.find("date_added"))
    # balanced bracket scan (treating \\" and \" as non-brackets)
    depth = 0
    end = -1
    i = start
    while i < len(big):
        c = big[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    block = big[start:end]
    # The embedded form uses \\" (one backslash + quote) for every structural quote.
    # Remove that single backslash to get raw JSON-ish text with real quotes.
    raw = block.replace('\\"', '"')
    return raw


def tolerant_parse_array(raw: str):
    """Parse a top-level JSON array of flat objects, tolerant of stray inner
    double-quotes inside string values."""
    # find each top-level object [start,end) by brace depth (ignore quotes)
    objs = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] == "[":
            i += 1
            continue
        if raw[i] == "{":
            depth = 0
            j = i
            while j < n:
                c = raw[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(raw[i : j + 1])
                        i = j + 1
                        break
                j += 1
        else:
            i += 1
    parsed = []
    for o in objs:
        parsed.append(parse_object(o))
    return parsed


def parse_object(o: str):
    """o is '{...}'. Parse known flat fields tolerantly."""
    # strip outer braces
    body = o[1:-1]
    result = {}
    # find all "key": ... by scanning; keys are always quoted identifiers.
    # key pattern: "wordchars": 
    pos = 0
    while True:
        m = re.search(r'"([a-zA-Z_][\w]*)"\s*:', body[pos:])
        if not m:
            break
        key = m.group(1)
        val_start = pos + m.end()
        # skip whitespace
        k = val_start
        while k < len(body) and body[k] in " \t":
            k += 1
        if k >= len(body):
            break
        if body[k] == '"':
            # string value: read until closing quote that is followed by , or }
            # find the structural closer: a " immediately followed by , or } (ignoring spaces)
            q = k + 1
            val = ""
            while q < len(body):
                c = body[q]
                if c == "\\":
                    val += body[q : q + 2]
                    q += 2
                    continue
                if c == '"':
                    # is this a structural closer? look ahead (skip spaces) for , or }
                    r = q + 1
                    while r < len(body) and body[r] in " \t":
                        r += 1
                    if r >= len(body) or body[r] in ",}":
                        break
                    else:
                        # inner quote -> keep as part of value
                        val += '"'
                        q += 1
                        continue
                val += c
                q += 1
            result[key] = val
            pos = q + 1
        else:
            # non-string value (number/bool/null)
            m2 = re.match(r"(-?\d+(?:\.\d+)?|true|false|null)", body[k:])
            if m2:
                result[key] = m2.group(1)
                pos = k + m2.end()
            else:
                pos = k + 1
    return result


def parse_tickerleague() -> pd.DataFrame:
    """Fetch the page and return a normalized DataFrame of changes."""
    r = requests.get(BASE, timeout=30, headers=H)
    raw = extract_array(r.text)
    data = tolerant_parse_array(raw)
    rows = []
    for d in data:
        eff = d.get("date") or d.get("date_added") or ""
        eff_iso = None
        try:
            eff_iso = dt.datetime.strptime(eff, "%B %d, %Y").date().isoformat()
        except Exception:
            try:
                eff_iso = dt.datetime.strptime(eff, "%Y-%m-%d").date().isoformat()
            except Exception:
                eff_iso = None
        rows.append(
            {
                "event_date": eff_iso,
                "event_date_raw": eff,
                "added_ticker": (d.get("symbol") or d.get("added_ticker") or "").upper() or None,
                "added_security": d.get("added_security") or None,
                "removed_ticker": (d.get("removed_ticker") or "").upper() or None,
                "removed_security": d.get("removed_security") or None,
                "reason": d.get("reason") or None,
                "index_type": d.get("index_type") or "sp500",
                "source": "tickerleague",
            }
        )
    rows.sort(key=lambda x: x["event_date"] or "", reverse=True)
    df = pd.DataFrame(rows)
    # event_date is a DATE column -> store as native date, not VARCHAR
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    return df


def main():
    df = parse_tickerleague()
    print(f"parsed rows: {len(df)}")
    if not df.empty:
        print("date range:", df["event_date"].dropna().min(), "->", df["event_date"].dropna().max())
        print("first:", df.iloc[0].to_dict())
    df.to_parquet("sp500_changes_tickerleague.parquet", index=False)
    print(f"wrote sp500_changes_tickerleague.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()

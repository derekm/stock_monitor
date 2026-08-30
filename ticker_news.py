#!/usr/bin/env python3
"""
ticker_news.py — Daily per-ticker news ingest + optional 3B desk note.

Polygon firehose (preferred), yfinance per-ticker fallback. Append-only
`ticker_news.parquet`. Optional `--notes` writes one-sentence press copy
into `ticker_news_notes.parquet` for the LLM brief (not a price call).

Usage:
    python ticker_news.py --save
    python ticker_news.py --save --days 2
    python ticker_news.py --notes --save --tickers AAPL,NVDA,META
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from analytics_common import DATA_DIR

OUT_NEWS = DATA_DIR / "ticker_news.parquet"
OUT_NOTES = DATA_DIR / "ticker_news_notes.parquet"
POLY_NEWS = "https://api.polygon.io/v2/reference/news"


def _polygon_key() -> str:
    k = os.environ.get("POLYGON_API_KEY", "").strip()
    if k:
        return k
    for p in (DATA_DIR / ".env", DATA_DIR.parent / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("POLYGON_API_KEY=") and not s.startswith("#"):
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _as_date(v) -> date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _write_dates(df: pd.DataFrame, path: Path, date_col: str) -> None:
    if df.empty:
        return
    df = df.copy()
    df[date_col] = [_as_date(v) for v in df[date_col]]
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    idx = tbl.schema.get_field_index(date_col)
    if idx >= 0:
        tbl = tbl.set_column(
            idx, date_col, pa.array(df[date_col].tolist(), type=pa.date32())
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)


def _load_news() -> pd.DataFrame:
    if not OUT_NEWS.exists():
        return pd.DataFrame()
    return pd.read_parquet(OUT_NEWS)


def fetch_polygon(days: int, api_key: str) -> pd.DataFrame:
    gte = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    url = POLY_NEWS
    params = {
        "limit": 100,
        "order": "desc",
        "sort": "published_utc",
        "published_utc.gte": gte,
        "apiKey": api_key,
    }
    next_url = url + "?" + urlencode(params)
    pages = 0
    while next_url and pages < 40:
        pages += 1
        r = requests.get(next_url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        for it in payload.get("results") or []:
            pub = it.get("published_utc") or ""
            d = _as_date(pub)
            if d is None:
                continue
            title = (it.get("title") or "").strip()
            src = ((it.get("publisher") or {}).get("name") or it.get("author") or "").strip()
            link = (it.get("article_url") or it.get("amp_url") or "").strip()
            aid = str(it.get("id") or hashlib.sha1((title + link).encode()).hexdigest()[:16])
            tickers = it.get("tickers") or []
            if not tickers:
                continue
            for t in tickers:
                t = str(t).upper().strip()
                if not t:
                    continue
                rows.append(
                    {
                        "ticker": t,
                        "published_date": d,
                        "published_utc": str(pub)[:25],
                        "headline": title[:300],
                        "source": src[:80],
                        "url": link[:400],
                        "article_id": aid,
                    }
                )
        nxt = payload.get("next_url")
        if not nxt:
            break
        next_url = nxt + ("&" if "?" in nxt else "?") + "apiKey=" + api_key
        time.sleep(0.12)
    return pd.DataFrame(rows)


def fetch_yfinance(tickers: list[str], days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for t in tickers:
        try:
            raw = yf.Ticker(t).news or []
        except Exception:
            raw = []
        for it in raw:
            content = it.get("content") if isinstance(it.get("content"), dict) else it
            title = (content.get("title") or it.get("title") or "").strip()
            link = ""
            cu = content.get("canonicalUrl") if isinstance(content, dict) else None
            if isinstance(cu, dict):
                link = cu.get("url") or ""
            link = link or it.get("link") or it.get("url") or ""
            pub = (
                content.get("pubDate")
                or content.get("displayTime")
                or it.get("providerPublishTime")
                or ""
            )
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(int(pub), tz=timezone.utc).isoformat()
            d = _as_date(pub)
            if d is None or d < cutoff.date():
                continue
            src = ""
            prov = content.get("provider") if isinstance(content, dict) else None
            if isinstance(prov, dict):
                src = prov.get("displayName") or ""
            src = src or it.get("publisher") or ""
            aid = hashlib.sha1((str(t) + title + str(link)).encode()).hexdigest()[:16]
            rows.append(
                {
                    "ticker": str(t).upper(),
                    "published_date": d,
                    "published_utc": str(pub)[:25],
                    "headline": title[:300],
                    "source": str(src)[:80],
                    "url": str(link)[:400],
                    "article_id": aid,
                }
            )
        time.sleep(0.05)
    return pd.DataFrame(rows)


def ingest(days: int, tickers: list[str] | None, save: bool) -> pd.DataFrame:
    key = _polygon_key()
    chunks = []
    if key:
        print("Polygon news firehose…")
        chunks.append(fetch_polygon(days, key))
    else:
        print("No POLYGON_API_KEY — yfinance fallback")
    if tickers:
        chunks.append(fetch_yfinance(tickers, days))
    elif not key:
        print("Need --tickers for yfinance fallback")
    new = pd.concat([c for c in chunks if c is not None and len(c)], ignore_index=True) if chunks else pd.DataFrame()
    if new.empty:
        print("No news rows")
        return new
    old = _load_news()
    if len(old):
        seen = set(zip(old["ticker"].astype(str), old["article_id"].astype(str)))
        keep = [
            (str(t), str(a)) not in seen
            for t, a in zip(new["ticker"], new["article_id"])
        ]
        new = new.loc[keep]
    print(f"new rows {len(new)} tickers {new['ticker'].nunique() if len(new) else 0}")
    if save and len(new):
        out = pd.concat([old, new], ignore_index=True) if len(old) else new
        _write_dates(out, OUT_NEWS, "published_date")
        print(f"Wrote {OUT_NEWS} ({len(out)} rows)")
    return new


_NAME_TOKS: dict[str, list[str]] | None = None


def _name_tokens(ticker: str) -> list[str]:
    global _NAME_TOKS
    if _NAME_TOKS is None:
        _NAME_TOKS = {}
        p = DATA_DIR / "monitored_stocks.parquet"
        if p.exists():
            m = pd.read_parquet(p)
            if "ticker" in m.columns and "name" in m.columns:
                for t, n in zip(m["ticker"].astype(str).str.upper(), m["name"].astype(str)):
                    toks = [t]
                    for w in n.replace(",", " ").split():
                        w = "".join(ch for ch in w if ch.isalpha())
                        if len(w) >= 4:
                            toks.append(w.upper())
                    _NAME_TOKS[t] = toks
    return _NAME_TOKS.get(ticker, [ticker])


def _headline_pack(news: pd.DataFrame, ticker: str, asof: date, lookback: int = 7) -> str | None:
    g = news[news["ticker"].astype(str) == ticker]
    if g.empty:
        return None
    lo = asof - timedelta(days=lookback)
    g = g.copy()
    g["_d"] = [_as_date(v) for v in g["published_date"]]
    g = g[g["_d"].notna() & (g["_d"] >= lo) & (g["_d"] <= asof)]
    if g.empty:
        return None
    g = g.rename(columns={"_d": "pub_d"})
    toks = _name_tokens(ticker)
    pat = "|".join(re.escape(t) for t in toks)
    named = g[g["headline"].astype(str).str.upper().str.contains(pat, na=False, regex=True)]
    if len(named):
        g = named
    g = g.sort_values(["pub_d", "published_utc"], ascending=False).drop_duplicates("headline").head(3)
    lines = [f"{d.isoformat()}: {h}" for d, h in zip(g["pub_d"], g["headline"])]
    return " | ".join(lines)


NEWS_SYSTEM = """You write one sentence of press context for a buy-side brief.
Use only the headlines. Do not invent numbers, prices, or filings.
Do not make a buy or sell call. Press is not leftover cash.
JSON keys: note (string, one sentence)."""


def _llm_note(headlines: str) -> str:
    from llama_cpp import Llama, LlamaGrammar
    from forecast_llm import MODEL_3B, _get_llm

    schema = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
    }
    llm, _ = _get_llm(MODEL_3B)
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    if hasattr(llm, "reset"):
        llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": NEWS_SYSTEM},
            {"role": "user", "content": headlines},
        ],
        max_tokens=60,
        temperature=0.3,
        grammar=grammar,
    )
    text = out["choices"][0]["message"]["content"]
    m = json.loads(text[text.find("{") : text.rfind("}") + 1])
    note = str(m.get("note") or "").strip()
    if not note:
        raise ValueError("empty note")
    return note[:220]


def write_notes(tickers: list[str] | None, save: bool, use_llm: bool) -> pd.DataFrame:
    news = _load_news()
    if news.empty:
        print("No ticker_news.parquet — ingest first")
        return pd.DataFrame()
    asof = date.today()
    names = tickers or sorted(news["ticker"].astype(str).unique())
    rows = []
    for i, t in enumerate(names, 1):
        pack = _headline_pack(news, t, asof)
        if not pack:
            continue
        if use_llm:
            try:
                note = _llm_note(pack)
            except Exception as e:
                print(f"  skip {t}: {e}", flush=True)
                continue
        else:
            note = pack
            if len(note) > 220:
                note = note[:217] + "..."
        rows.append({"ticker": t, "as_of_date": asof, "n_headlines": pack.count("|") + 1, "news_note": note})
        print(f"{i}/{len(names)} {t} {note[:80]}", flush=True)
    out = pd.DataFrame(rows)
    if save and len(out):
        old = pd.read_parquet(OUT_NOTES) if OUT_NOTES.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["ticker"].astype(str).isin(set(out["ticker"]))]
            out = pd.concat([old, out], ignore_index=True)
        _write_dates(out, OUT_NOTES, "as_of_date")
        print(f"Wrote {OUT_NOTES} ({len(out)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--notes", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="Store headline pack, skip 3B")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    if args.notes:
        if not OUT_NEWS.exists():
            ingest(args.days, tickers, args.save)
        write_notes(tickers, args.save, use_llm=not args.no_llm)
    else:
        ingest(args.days, tickers, args.save)


if __name__ == "__main__":
    main()

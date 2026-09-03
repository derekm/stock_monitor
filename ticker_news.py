#!/usr/bin/env python3
"""
ticker_news.py — Daily per-ticker news ingest + 3B mention/press pipeline.

Polygon firehose (preferred), yfinance per-ticker fallback. Append-only
`ticker_news.parquet`. `--mentions` extracts per-article mentions with the
3B; `--press` renders one gated sentence per ticker for the LLM brief (not
a price call). Both 3B stages run on Intel Iris Xe (Vulkan, `.venv-xpu`),
not MX550.

Usage:
    python ticker_news.py --save
    python ticker_news.py --save --days 2
    python ticker_news.py --press --save --tickers AAPL,NVDA,META
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
OUT_ARTICLES = DATA_DIR / "ticker_news_articles.parquet"
OUT_MENTIONS = DATA_DIR / "ticker_news_mentions.parquet"
OUT_PRESS = DATA_DIR / "ticker_news_press.parquet"
BUCKET = DATA_DIR / "news_articles"
MODEL_3B = Path(r"C:\Users\derek\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf")
XPU_PYTHON = Path(r"C:\Users\derek\src\stockmagic\.venv-xpu\Scripts\python.exe")
DOCLING_PY = Path(r"C:\Users\derek\src\docling\.venv\Scripts\python.exe")
DOCLING_HTML = Path(r"C:\Users\derek\src\docling\html_to_md.py")
NEWS_CTX_MAX = 32768  # llama-cli -c ceiling; live n_ctx is article-sized
_news_llm = None
_news_ctx = 0
_GGML_TYPE_Q8_0 = 8  # llama-cli -ctk q8_0 -ctv q8_0
POLY_NEWS = "https://api.polygon.io/v2/reference/news"
_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}


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
        # Polygon news rate-limits hard (429); back off and retry the page
        # instead of aborting the whole firehose pull.
        for attempt in range(6):
            r = requests.get(next_url, timeout=30)
            if r.status_code == 429:
                time.sleep(min(8 * (2 ** attempt), 120))
                continue
            break
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
        extract_articles(save=True, recent_days=days)
    elif save:
        extract_articles(save=True, recent_days=days)
    return new


def _object_key(published, article_id: str, ext: str = "md") -> str:
    d = _as_date(published) or date.today()
    return f"{d.year:04d}/{d.month:02d}/{d.day:02d}/{article_id}.{ext}"


def bucket_put_text(key: str, text: str) -> Path:
    path = BUCKET / key
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def bucket_exists(key: str) -> bool:
    return (BUCKET / key).is_file()


def bucket_read(published, article_id: str) -> str:
    """Text body. .md first; leftover .json objects are a one-time fallback."""
    md = BUCKET / _object_key(published, article_id, "md")
    if md.is_file():
        return md.read_text(encoding="utf-8", errors="replace")
    js = BUCKET / _object_key(published, article_id, "json")
    if js.is_file():
        try:
            return str(json.loads(js.read_text(encoding="utf-8")).get("body") or "")
        except Exception:
            return ""
    return ""


def _extract_html(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    ps = []
    for p in soup.find_all("p"):
        s = " ".join(p.get_text(" ", strip=True).split())
        if len(s) >= 40:
            ps.append(s)
    skip = ("cookie", "subscribe", "sign up for", "advertisement")
    ps = [p for p in ps if not any(k in p.lower()[:48] for k in skip)]
    text = "\n\n".join(ps)
    if len(text) >= 200:
        return text[:80000]
    md = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
        "meta", attrs={"name": "description"}
    )
    if md and md.get("content"):
        return str(md["content"]).strip()[:80000]
    return text[:80000]


def _docling_body(html: str) -> str:
    """Docling HTML backend → text. Empty string means fall back to bs4."""
    if not DOCLING_PY.is_file() or not DOCLING_HTML.is_file():
        return ""
    import subprocess
    import tempfile

    html_bytes = html.encode("utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "article.html"
        dst = Path(td) / "article.md"
        src.write_bytes(html_bytes)
        env = os.environ.copy()
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        env.setdefault("TORCHINDUCTOR_DISABLE", "1")
        r = subprocess.run(
            [str(DOCLING_PY), str(DOCLING_HTML), str(src), str(dst)],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        if r.returncode != 0 or not dst.is_file():
            return ""
        return dst.read_text(encoding="utf-8", errors="replace").strip()


def fetch_article(url: str) -> tuple[str, str]:
    """Return (status, body). status is ok, empty, http_N, or error."""
    last = "error:ReadTimeout"
    for attempt in range(3):
        try:
            r = requests.get(
                url, headers=_UA, timeout=(8, 20), allow_redirects=True
            )
        except Exception as e:
            last = f"error:{type(e).__name__}"
            time.sleep(0.4 * (attempt + 1))
            continue
        if r.status_code != 200:
            return f"http_{r.status_code}", ""
        body = _docling_body(r.text) or _extract_html(r.text)
        if len(body) < 80:
            return "empty", body
        return "ok", body
    return last, ""


def extract_articles(save: bool = True, limit: int | None = None,
                     recent_days: int | None = None) -> pd.DataFrame:
    news = _load_news()
    if news.empty:
        print("No ticker_news.parquet — ingest first")
        return pd.DataFrame()
    uniq = news.drop_duplicates("article_id")
    if recent_days is not None:
        # Daily runs must NOT re-crawl the whole unfetched backlog (1,899
        # bodies were stuck behind globenewswire/zacks timeouts, extending
        # ticker_news past its window on 2026-09-03). Only articles inside
        # the ingest window get body fetches; older gaps stay for a
        # dedicated backfill (`--extract` without a window).
        cutoff = date.today() - timedelta(days=max(0, int(recent_days)))
        before = len(uniq)
        uniq = uniq[uniq["published_date"].map(_as_date) >= cutoff]
        print(f"  extract window: {cutoff.isoformat()}+ ({len(uniq)} of {before} articles)", flush=True)
    rows = []
    n = 0
    for rec in uniq.itertuples(index=False):
        aid = str(rec.article_id)
        url = str(rec.url or "")
        pub = _as_date(rec.published_date)
        if not url or pub is None:
            continue
        key = _object_key(pub, aid, "md")
        if bucket_exists(key) or bucket_exists(_object_key(pub, aid, "json")):
            continue
        n += 1
        if limit is not None and n > limit:
            break
        status, body = fetch_article(url)
        if save and status == "ok" and body:
            bucket_put_text(key, body)
        rows.append({
            "article_id": aid,
            "url": url,
            "headline": str(rec.headline or "")[:300],
            "source": str(rec.source or "")[:80],
            "published_date": pub,
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "n_chars": len(body),
            "key": key,
        })
        print(f"  {status} {len(body):5d} {url[:70]}", flush=True)
        time.sleep(0.15)
    idx = pd.DataFrame(rows)
    if save and len(idx):
        old = pd.read_parquet(OUT_ARTICLES) if OUT_ARTICLES.exists() else pd.DataFrame()
        if len(old):
            seen = set(old["article_id"].astype(str))
            idx = idx[~idx["article_id"].astype(str).isin(seen)]
            idx = pd.concat([old, idx], ignore_index=True) if len(idx) else old
        _write_dates(idx, OUT_ARTICLES, "published_date")
        print(f"Wrote {OUT_ARTICLES} ({len(idx)} rows); bucket {BUCKET}")
    else:
        print(f"extracted {len(idx)} new objects")
    return idx


MENTION_SYSTEM = """You list companies this article actually discusses.
Return JSON only. Each mention is one company the article covers in substance, not a passing ticker tag.
Do not invent tickers. If the company has no listed ticker, leave ticker empty.
summary is one sentence about that company in this article, from that company's side.
JSON keys: mentions (array of {company, ticker, summary})."""


def _pin_intel_vulkan() -> None:
    os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
    os.environ.setdefault("GGML_VK_PREFER_HOST_MEMORY", "1")


def _n_ctx_for_prompts(*parts: str) -> int:
    """llama-cli -c sized to system + user (article/doctags), cap NEWS_CTX_MAX."""
    n_chars = sum(len(p or "") for p in parts)
    need = max(2048, n_chars // 3 + 384)
    need = min(NEWS_CTX_MAX, need)
    return ((need + 255) // 256) * 256


def _get_news_llm(n_ctx: int | None = None):
    """3B ≡ llama-cli -ngl 99 -b 256 -ub 256 -ctk q8_0 -ctv q8_0 --flash-attn -t 6.
    n_ctx is article-sized (llama-cli -c). Refuses the CUDA wheel."""
    global _news_llm, _news_ctx
    _pin_intel_vulkan()
    import llama_cpp
    from llama_cpp import Llama

    lib = Path(llama_cpp.__file__).parent
    vulkan = (lib / "lib" / "ggml-vulkan.dll").is_file() or (lib / "ggml-vulkan.dll").is_file()
    if not vulkan:
        raise SystemExit(
            "News LLM is Intel Vulkan, not MX550. "
            f"Use {XPU_PYTHON}"
        )
    ctx = int(n_ctx) if n_ctx else 2048
    if _news_llm is not None and ctx > _news_ctx:
        del _news_llm
        _news_llm = None
    if _news_llm is None:
        print(
            f"Initializing 3B on Intel Iris Xe (Vulkan0) n_ctx={ctx} "
            f"ngl=99 b=256 ctk=q8_0 t=6...",
            flush=True,
        )
        kwargs = dict(
            model_path=str(MODEL_3B),
            n_gpu_layers=99,
            n_ctx=ctx,
            n_batch=256,
            n_ubatch=256,
            n_threads=6,
            flash_attn=True,
            type_k=_GGML_TYPE_Q8_0,
            type_v=_GGML_TYPE_Q8_0,
            chat_format="llama-3",
            verbose=False,
        )
        try:
            _news_llm = Llama(**kwargs)
        except Exception as e:
            print(f"  q8 KV + flash_attn failed ({e}); retry type_v default", flush=True)
            kwargs.pop("type_v", None)
            try:
                _news_llm = Llama(**kwargs)
            except Exception as e2:
                print(f"  flash_attn failed ({e2}); retry flash_attn=False", flush=True)
                kwargs["flash_attn"] = False
                kwargs["type_v"] = _GGML_TYPE_Q8_0
                _news_llm = Llama(**kwargs)
        _news_ctx = ctx
    return _news_llm


def _llm_mentions(headline: str, body: str, hint: str) -> list[dict]:
    from llama_cpp import LlamaGrammar

    schema = {
        "type": "object",
        "properties": {
            "mentions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "ticker": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["company", "ticker", "summary"],
                },
            }
        },
        "required": ["mentions"],
    }
    user = f"Headline: {headline}\n"
    if hint:
        user += f"Polygon tagged (hint only): {hint}\n"
    user += "Article:\n" + (body or "")
    llm = _get_news_llm(_n_ctx_for_prompts(MENTION_SYSTEM, user))
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    if hasattr(llm, "reset"):
        llm.reset()
    cap = max(2000, (_news_ctx - 1024) * 3)
    if len(user) > cap:
        user = user[:cap]
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": MENTION_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=180,
        temperature=0.2,
        grammar=grammar,
    )
    text = out["choices"][0]["message"]["content"]
    m = json.loads(text[text.find("{") : text.rfind("}") + 1])
    return list(m.get("mentions") or [])


PRESS_SYSTEM = """You are a financial news summarizer.
Input: a list of per-article summaries about ONE company.
Output: ONE sentence — the press line for that company's brief.
Rules:
- Synthesize; do not list.
- No hedging words ("may", "could", "might").
- No meta commentary ("the article says").
- Facts only: product, deal, guidance, earnings, M&A, regulatory, technical.
- If nothing material, return empty string."""


def _llm_press_line(ticker: str, summaries: list[str]) -> str:
    from llama_cpp import LlamaGrammar

    schema = {
        "type": "object",
        "properties": {
            "press_line": {"type": "string", "description": "One sentence news sentiment blurb for the brief"}
        },
        "required": ["press_line"],
    }
    user = "Company: " + ticker + "\nArticle summaries:\n" + "\n".join(f"- {s}" for s in summaries) + '\n\nReturn JSON: {"press_line": "..."} (empty string if no material news)'
    llm = _get_news_llm(_n_ctx_for_prompts(PRESS_SYSTEM, user))
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    if hasattr(llm, "reset"):
        llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": PRESS_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
        temperature=0.1,
        grammar=grammar,
    )
    text = out["choices"][0]["message"]["content"]
    m = json.loads(text[text.find("{") : text.rfind("}") + 1])
    press_line = str(m.get("press_line") or "").strip()
    return press_line


def _known_tickers() -> set[str]:
    s: set[str] = set()
    p = DATA_DIR / "monitored_stocks.parquet"
    if p.exists():
        m = pd.read_parquet(p)
        if "ticker" in m.columns:
            s |= set(m["ticker"].astype(str).str.upper())
    news = _load_news()
    if len(news):
        s |= set(news["ticker"].astype(str).str.upper())
    return s


def write_mentions(save: bool, use_llm: bool, limit: int | None = None) -> pd.DataFrame:
    if not OUT_ARTICLES.exists():
        print("No article index — extract first")
        return pd.DataFrame()
    arts = pd.read_parquet(OUT_ARTICLES)
    arts = arts[arts["status"].astype(str) == "ok"]
    if arts.empty:
        print("No extracted bodies")
        return pd.DataFrame()
    done = set()
    if OUT_MENTIONS.exists():
        old_m = pd.read_parquet(OUT_MENTIONS)
        done = set(old_m["article_id"].astype(str))
    else:
        old_m = pd.DataFrame()
    news = _load_news()
    rows = []
    n = 0
    for rec in arts.itertuples(index=False):
        aid = str(rec.article_id)
        if aid in done:
            continue
        n += 1
        if limit is not None and n > limit:
            break
        body = bucket_read(rec.published_date, aid)
        if not body:
            continue
        headline = str(rec.headline or "")
        hint = ""
        if len(news):
            tags = news.loc[news["article_id"].astype(str) == aid, "ticker"].astype(str).unique().tolist()
            hint = ",".join(tags[:12])
        if not use_llm:
            continue
        try:
            mentions = _llm_mentions(headline, body, hint)
            print(json.dumps(mentions, ensure_ascii=False)[:1200], flush=True)
        except Exception as e:
            print(f"  skip {aid[:12]}: {e}", flush=True)
            continue
        pub = _as_date(rec.published_date)
        for m in mentions:
            t = str(m.get("ticker") or "").strip().upper()
            if t in {"", "NONE", "N/A", "NA", "NULL"}:
                t = ""
            summ = str(m.get("summary") or "").strip()
            if not summ:
                continue
            rows.append({
                "article_id": aid,
                "url": str(getattr(rec, "url", "") or "")[:400],
                "ticker": t,
                "company": str(m.get("company") or t)[:80],
                "summary": summ[:400],
                "published_date": pub,
            })
        print(f"  mentions {aid[:12]} n={len(rows)}", flush=True)
    out = pd.DataFrame(rows)
    if save and len(out):
        if len(old_m):
            out = pd.concat([old_m, out], ignore_index=True)
        _write_dates(out, OUT_MENTIONS, "published_date")
        print(f"Wrote {OUT_MENTIONS} ({len(out)} rows)")
    return out


def write_press(save: bool, use_llm: bool) -> pd.DataFrame:
    """Render per-ticker press lines from mentions."""
    if not OUT_MENTIONS.exists():
        print("No mentions — run --mentions first")
        return pd.DataFrame()
    mentions = pd.read_parquet(OUT_MENTIONS)
    if mentions.empty:
        print("No mention rows")
        return pd.DataFrame()

    # Group by ticker (skip blank)
    tickers = [t for t in mentions["ticker"].unique() if t and t.strip()]
    print(f"Rendering press lines for {len(tickers)} tickers...")

    rows = []
    for ticker in tickers:
        t_men = mentions[mentions["ticker"] == ticker]
        summaries = t_men["summary"].tolist()
        if not summaries:
            continue
        if use_llm:
            try:
                press_line = _llm_press_line(ticker, summaries)
            except Exception as e:
                print(f"  skip {ticker}: {e}", flush=True)
                continue
        else:
            press_line = " | ".join(summaries[:3])
        rows.append({"ticker": ticker, "press_line": press_line, "n_mentions": len(summaries), "as_of_date": date.today()})
        if press_line:
            print(f"  {ticker}: {press_line}")
        else:
            print(f"  {ticker}: (empty)")

    out = pd.DataFrame(rows)
    if save and len(out):
        old = pd.read_parquet(OUT_PRESS) if OUT_PRESS.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["ticker"].astype(str).isin(set(out["ticker"]))]
            out = pd.concat([old, out], ignore_index=True)
        _write_dates(out, OUT_PRESS, "as_of_date")
        print(f"Wrote {OUT_PRESS} ({len(out)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--extract", action="store_true", help="Fetch article bodies into news_articles/")
    ap.add_argument("--mentions", action="store_true", help="LLM mention JSON per article (needs GPU)")
    ap.add_argument("--press", action="store_true", help="Render per-ticker press lines from mentions (needs GPU)")
    ap.add_argument("--limit", type=int, default=0, help="Cap new extracts/mentions (0 = all)")
    ap.add_argument("--no-llm", action="store_true", help="Skip 3B")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    lim = args.limit or None
    if (args.mentions or args.press) and not args.no_llm:
        _pin_intel_vulkan()
    if args.extract:
        extract_articles(save=args.save, limit=lim)
        return
    if args.mentions:
        write_mentions(save=args.save, use_llm=not args.no_llm, limit=lim)
        return
    if args.press:
        write_press(save=args.save, use_llm=not args.no_llm)
        return
    ingest(args.days, tickers, args.save)


if __name__ == "__main__":
    main()

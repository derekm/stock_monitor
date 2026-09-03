# ticker_news.py — daily per-ticker press ingest + article bucket + notes

Polygon firehose (preferred) or yfinance fallback. Article bodies live as
text (`news_articles/YYYY/MM/DD/{article_id}.md`) via Docling's HTML backend.
Mentions are LLM JSON per article so one story can cover every company it
actually names. A second LLM call folds those summaries into one brief sentence.

## Why it exists

The forecast brief is operating facts. Press is a dated sidecar — not leftover
cash, not a buy call. Polygon ticker tags leak (one Fool story tagged AAPL).
Mentions come from the article, not the tag list.

## Object store

Layout is S3-compatible so the folder can be synced later:

`news_articles/YYYY/MM/DD/{article_id}.md`

Plain UTF-8 article text (Docling HTML → `export_to_text`, bs4 fallback).
Metadata lives in `ticker_news_articles.parquet`, not in the object.
Do not commit `news_articles/`. Mentions stay JSON arrays in parquet.

## Usage

```bash
python ticker_news.py --save --days 2          # firehose + extract bodies
python ticker_news.py --extract --save         # bodies only (resume)
python ticker_news.py --mentions --save        # 3B JSON mentions on Intel Vulkan
python ticker_news.py --press --save           # per-ticker press lines from mentions
```

`--mentions` / `--press` without `--no-llm` load the 3B on Intel Iris Xe
(Vulkan0, `.venv-xpu`). Equivalent to:

`llama-cli -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl 99 -c <sys+user> -ub 256 -b 256 -ctk q8_0 -ctv q8_0 --flash-attn -t 6`

`-c` is system (extraction goals + JSON schema) plus user (article text), cap 32768.
Pin `GGML_VK_VISIBLE_DEVICES=0`.

```bash
C:/Users/derek/src/stockmagic/.venv-xpu/Scripts/python.exe ticker_news.py --mentions --save
C:/Users/derek/src/stockmagic/.venv-xpu/Scripts/python.exe ticker_news.py --press --save
```

CUDA `.venv` refuses the news LLM (no `ggml-vulkan.dll`). Measured 2026-08-30
on this 3B, n_ctx=2048, -b 256 -t 6, count-1-to-16 (32 tok): **CPU ngl=0
4.15s / ~7.5 tok/s mix**; **Xe ngl=99 q8 KV + FA 6.61s / ~4.8 tok/s mix**.
CPU is faster; Xe is for not stealing forecast cores/MX550. First Xe 0.26 tok/s
was n_ctx=512 / n_batch=128 without q8 KV — not this CLI. Do not set
`GGML_VULKAN_DISABLE_F16` (Iris Xe fp16 works; that flag is f32 shaders, slower).
Granite-Docling PDF VLM stays in `C:\Users\derek\src\docling\.venv` for
`~/research` PDFs. News HTML uses the same venv's **HTML backend**
(`html_to_md.py`, ~1.5s/page) — not the PDF VLM. Bodies are `.md` text, not
JSON. Mention output is still a JSON array.

## Outputs

- `ticker_news.parquet` — firehose index (headline, url, polygon tickers)
- `news_articles/` — article objects
- `ticker_news_articles.parquet` — object index
- `ticker_news_mentions.parquet` — article_id (FK), url, ticker, company, summary. Check in.
  Not the body.
- `ticker_news_press.parquet` — ticker, press_line, n_mentions, as_of_date

All 3B stages (`--mentions`, `--press`) run on Intel Iris Xe
Vulkan (`.venv-xpu`). The daily DAG runs them serially after Docling and
before `llm_forecast`: firehose+extract → mentions → press → forecast.
Notes (`--notes`, headline-only desk notes) were removed — mentions +
press replaced them; the LLM brief reads `press_line` only.

## Related

- [forecast_llm.py](../forecast_llm.py) — gated Press line
- [filings_sentiment.md](filings_sentiment.md) — 8-K lexicon, not press

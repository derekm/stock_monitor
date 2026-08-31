# ticker_news.py — daily per-ticker press ingest + article bucket + notes

Polygon firehose (preferred) or yfinance fallback. Article bodies live in an
S3-key object store (`news_articles/YYYY/MM/DD/{article_id}.json`). Mentions
are LLM JSON per article so one story can cover every company it actually
names. A second LLM call folds those summaries into one brief sentence.

## Why it exists

The forecast brief is operating facts. Press is a dated sidecar — not leftover
cash, not a buy call. Polygon ticker tags leak (one Fool story tagged AAPL).
Mentions come from the article, not the tag list.

## Object store

Layout is S3-compatible so the folder can be synced later:

`news_articles/YYYY/MM/DD/{article_id}.json`

Each object: article_id, url, headline, source, published_date, fetched_utc,
status, n_chars, body. Index: `ticker_news_articles.parquet` (no body).
Do not commit `news_articles/`.

## Usage

```bash
python ticker_news.py --save --days 2          # firehose + extract bodies
python ticker_news.py --extract --save         # bodies only (resume)
python ticker_news.py --mentions --save        # 3B JSON mentions on Intel Vulkan
python ticker_news.py --notes --save           # mention packs → one-sentence notes
python ticker_news.py --notes --save --no-llm  # packs only, no GGUF
```

`--mentions` / `--notes` without `--no-llm` load the 3B on Intel Iris Xe
(Vulkan0, `.venv-xpu`). Equivalent to:

`llama-cli -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl 99 -c <sys+user> -ub 256 -b 256 -ctk q8_0 -ctv q8_0 --flash-attn -t 6`

`-c` is system (extraction goals + JSON schema) plus user (article text), cap 32768.
Forecast stays on CUDA MX550. Pin `GGML_VK_VISIBLE_DEVICES=0`.

```bash
C:/Users/derek/src/stockmagic/.venv-xpu/Scripts/python.exe ticker_news.py --mentions --save
C:/Users/derek/src/stockmagic/.venv-xpu/Scripts/python.exe ticker_news.py --notes --save
```

CUDA `.venv` refuses the news LLM (no `ggml-vulkan.dll`). Measured 2026-08-30
on this 3B, n_ctx=2048, -b 256 -t 6, count-1-to-16 (32 tok): **CPU ngl=0
4.15s / ~7.5 tok/s mix**; **Xe ngl=99 q8 KV + FA 6.61s / ~4.8 tok/s mix**.
CPU is faster; Xe is for not stealing forecast cores/MX550. First Xe 0.26 tok/s
was n_ctx=512 / n_batch=128 without q8 KV — not this CLI. Do not set
`GGML_VULKAN_DISABLE_F16` (Iris Xe fp16 works; that flag is f32 shaders, slower).
Granite-Docling is `C:\Users\derek\src\docling\.venv` for `~/research` PDFs
(2–7 min/paper). It is not the news HTML path and does not replace mention JSON.

## Outputs

- `ticker_news.parquet` — firehose index (headline, url, polygon tickers)
- `news_articles/` — article objects
- `ticker_news_articles.parquet` — object index
- `ticker_news_mentions.parquet` — article_id (FK), url, ticker, company, summary. Check in.
  Not the body.
- `ticker_news_notes.parquet` — ticker, as_of_date, news_note

## Related

- [forecast_llm.py](../forecast_llm.py) — gated Press line
- [filings_sentiment.md](filings_sentiment.md) — 8-K lexicon, not press

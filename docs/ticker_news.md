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
python ticker_news.py --mentions --save        # 3B JSON mentions — not during forecast
python ticker_news.py --notes --save           # mention packs → one-sentence notes
python ticker_news.py --notes --save --no-llm  # packs only, no GGUF
```

`--mentions` / `--notes` without `--no-llm` load the 3B on the MX550 — do not
overlap a `forecast_llm` run.

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

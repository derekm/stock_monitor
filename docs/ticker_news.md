# ticker_news.py — daily per-ticker press ingest + optional 3B note

Polygon firehose (preferred) or yfinance fallback. Append-only headlines,
then a one-sentence press line for the LLM desk brief.

## Why it exists

The forecast brief is operating facts. Headlines are a separate, dated
sidecar — not leftover cash, not a buy call. Python owns the articles;
`--notes` asks the 3B for one sentence. Omit the line when the sidecar is
empty.

## Usage

```bash
python ticker_news.py --save --days 2
python ticker_news.py --notes --save
python ticker_news.py --notes --save --no-llm --tickers AAPL,NVDA
```

`--notes` loads the 3B on the MX550 — do not overlap a `forecast_llm` run.

## Outputs

- `ticker_news.parquet` — ticker, published_date (date32), published_utc, headline, source, url, article_id
- `ticker_news_notes.parquet` — ticker, as_of_date (date32), n_headlines, news_note

## Related

- [forecast_llm.py](../forecast_llm.py) — gated `Press (not a reason to own):` line
- [filings_sentiment.md](filings_sentiment.md) — 8-K lexicon, not press

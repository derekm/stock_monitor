# filings_sentiment.py

Lightweight lexicon sentiment on recent SEC 8-K filings (alternative-data
MVP, no external NLP dependency).

## Why it exists (rationale)

Closes the "alternative data — NLP on filings" TODO at the honest MVP level:
pull the most recent 8-K (current report) for each ticker from SEC EDGAR,
strip HTML/scripts, and score the body with Loughran-McDonald-style financial
word lists (negative / positive / forward-looking).

## Method

- CIK via `company_tickers.json`; recent 8-K accessions via the submissions
  API; document body fetched from the Archives (longest .htm — the exhibit
  press release carries the substance; the SGML wrapper is discarded).
- Lexicon scoring: case-insensitive word-boundary counts; `score_per_1k` =
  (n_neg − n_pos) / token_count × 1000.
- SEC rate limits respected (0.11s sleep).

Honesty notes: 8-Ks are often boilerplate (appointments, notices) — near-zero
sentiment IS the signal (no news beats bad news). Lexicon matching is
unlemmatized.

## Usage

```bash
python filings_sentiment.py --save
python filings_sentiment.py --save --tickers AAPL,MSFT --limit 3
```

## Outputs

- `filings_sentiment.csv` — (ticker, filing_date, form, n_neg, n_pos,
  n_fwd, score_per_1k).

## Related programs

- `backfill_edgar.py` — the EDGAR/XBRL fundamentals pipeline
- `options_skew.py` — sibling alternative-data snapshotter

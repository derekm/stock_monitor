# implied_r_screen.py

**Implied cost-of-capital screen** — Ohlson & Rueangsuwan (2026), *Formal Equity Valuation: Overview and Limits* (SSRN 6280638).

## Why it exists (rationale)

Ohlson & Rueangsuwan argue the most defensible use of valuation formulas is **not** estimating P (fragile r and g — any `1/(r−g)` is explosive) but **inferring r** from the current price and fundamentals, then asking whether the market's implied cost of capital looks high (cheap) or low (expensive). This script implements that screen with zero analyst forecasts.

## Formula

RIV reduced form with `g = r/2` (the paper's bounded-rationality default):

    P = −BV + 2·X(1)/r   →   r = 2·X(1)/(P + BV)

With `X(1) = ROE·BV` (expected next-period earnings on current book) and `BV = P/(P/B)`:

    r_implied = 2·ROE/(P/B + 1)

One observable per ticker: ROE and P/B from `fundamentals.parquet`, price from `daily_prices.parquet`.

## Usage

    python implied_r_screen.py                 # print the screen
    python implied_r_screen.py --save          # write implied_r_screen.csv/.parquet
    python implied_r_screen.py --min-cap 10    # only names > $10B market cap
    python implied_r_screen.py --top 50        # rows per verdict bucket

## Outputs

| File | Contents |
|------|----------|
| `implied_r_screen.csv` | ticker, price, bvps, pb_ratio, roe, implied_r_pct, fwd_pe_bench, verdict, mktcap_b, ev_ebitda, roic, r_gt_roe, pb_lt_1, triplet_ok, as_of |
| `implied_r_screen.parquet` | same, parquet |

Verdict thresholds (paper's worked example uses r = 9%): CHEAP ≥ 12%, Fair 7–10%, EXPENSIVE ≤ 6%.

## Related

- `docs/ohlson_rueangsuwan_formal_equity_valuation_2026.md` (research library)
- `preferred_metrics.py` (trifecta/quality screen — this is the cross-check, not a standalone signal)
- Dashboard: Value tab `tbl-implied-r` (live DuckDB query)

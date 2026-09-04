# universal_portfolio.py — Cover Universal Portfolio

One-line: model-free online portfolio that matches the best constant-rebalanced
portfolio in hindsight up to O(log n) regret, with zero statistical assumptions
and zero lookahead.

## Why it exists (rationale)

The CRP oracle S*_n (best constant-rebalanced portfolio ex post) is the central
object behind kelly.py's leverage space, ERC/HRP, and the Bogle TMI/BPI mixes.
Cover (1991) built the *no-lookahead competitor* to that oracle: hold, each day,
the performance-weighted average of ALL CRPs in the simplex. It needs no model,
no forecasts, no shorts; the Dirichlet(1/2) prior is minimax-optimal and the
regret vs hindsight is exactly ((m-1)/2) log(n+1) (Ordentlich-Cover 1998).
It is the honest baseline the forecast layer must beat — "what you get from the
market alone, without believing anything."

## Theory / math

- Day-k weight: b_hat_k = ∫ b·S_{k-1}(b) dμ(b) / ∫ S_{k-1}(b) dμ(b), μ = Dirichlet(1/2).
- Wealth telescopes to Ŝ_n = ∫ S_n(b) dμ(b) — a plain integral over the simplex.
- m=2: exact via Cover eq. (128) Q-recursion, O(n²), self-checked by the
  telescope identity Ŝ_n == Σ_l Q_n(l) (asserted every run).
- m≥2: simplex Monte-Carlo mixture (Kalai-Vempala sampling). Validated vs the
  exact engine to < 0.05% on synthetic 2-asset data.

## Usage

```
# research: 400 shared block-bootstrap paths (block 21, seed 0 — same paths as
# cppi_backtest.py item 14), universal vs erc/hrp/cppi_m3/vincent_ls
python universal_portfolio.py paths --paths 400 --save

# personal book: daily universal weights for the 9 book names, Cover sec.10
# cost gate (rebalance only when log-wealth gain > log(1+cost))
python universal_portfolio.py book --save [--tickers ...] [--cost 0.001]

# engine validation (exact vs MC on synthetic 2-asset data)
python universal_portfolio.py validate --samples 100000
```

Standard `cli_common` flags do not apply (standalone research tool).

## Outputs

| Output | Producer | Family |
|---|---|---|
| `universal_paths.parquet` | `universal_portfolio.py paths` | Other |
| `universal_book_weights.parquet` | `universal_portfolio.py book` | Other |

- `universal_paths.parquet`: book × (terminal, maxdd, underwater_days) over 400
  bootstrap paths — same stats schema as `cppi_paths.parquet` (item 14).
- `universal_book_weights.parquet`: date × ticker × weight (cost-gated) for the
  personal book. Derived panel — the script only READS `daily_prices/`.

## Related programs

- `cppi_backtest.py` — item 14 (CPPI vs ERC vs HRP vs Vince LS; same paths)
- `kelly.py` — Vince leverage space / log-optimal RS (the CRP-oracle family)
- `build_bogle_funds.py` — TMI/BPI, the 2-asset substrate for `paths`
- docs/cover_universal_portfolio.md — the research brief (theory + exact tables)

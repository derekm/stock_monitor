# universal_portfolio.py — Cover Universal Portfolio (reusable library)

One-line: model-free online portfolio that matches the best constant-rebalanced
portfolio in hindsight up to O(log n) regret, with zero statistical assumptions
and zero lookahead — provided as importable objects plus a thin CLI.

## Why it exists (rationale)

The CRP oracle S*_n (best constant-rebalanced portfolio ex post) is the central
object behind kelly.py's leverage space, ERC/HRP, and the Bogle TMI/BPI mixes.
Cover (1991) built the *no-lookahead competitor* to that oracle: hold, each day,
the performance-weighted average of ALL CRPs in the simplex. It needs no model,
no forecasts, no shorts; the Dirichlet(1/2) prior is minimax-optimal and the
regret vs hindsight is exactly ((m-1)/2) log(n+1) (Ordentlich-Cover 1998).
It is the honest baseline the forecast layer must beat — "what you get from the
market alone, without believing anything."

## Library (importable — not just a CLI)

- `UniversalEngine` — engines over an (n, m) return matrix, numpy-only, no IO:
  - `exact_2asset(r1, r2)` — Cover eq. (128) Q-telescope, O(n²), EXACT for m=2;
    the telescope identity Ŝ_n == Σ_l Q_n(l) is ASSERTED every run
  - `mc_simplex(R, ...)` — Kalai-Vempala simplex sampling for m≥2, validated
    vs exact to <0.05% on synthetic 2-asset data
  - `run(R, ...)` — m==2 picks exact, m>2 picks MC; returns PortfolioResult
- `PortfolioResult` — weights (n,m), wealth (n+1,), `apply_gate(cost)` (Cover
  §10: rebalance only when ΔlnW > ln(1+cost), mutates in place with
  `inplace=True` or returns a copy), `stats()` (terminal/maxDD/underwater),
  `wealth_from(weights)`.
- `RegimeStates` — tape-level HMM side info (Cover & Ordentlich 1996): 3-state
  Gaussian HMM on the tape's own features (mkt_ret, vol21, avg pairwise corr —
  the SAME features as `hmm_regime_detection.py` via its `build_features`).
  Fit on the ORIGINAL tape only (never a resampled path — no lookahead).
  `side_portfolio(R)` = state-conditional universal portfolio (per-state
  subsequence engine; sparse-state days fall back to the plain universal
  weight so a bootstrap path with <2 days of a state can't zero out).
- `SizingPlan` — target weights (latest gated universal) vs
  `portfolio_holdings.parquet` → per-name delta weight/notional/shares at last
  close, gate-honoring. Handles the percent-form `weight` column.

## Theory / math

- Day-k weight: b_hat_k = ∫ b·S_{k-1}(b) dμ(b) / ∫ S_{k-1}(b) dμ(b), μ = Dirichlet(1/2).
- Wealth telescopes to Ŝ_n = ∫ S_n(b) dμ(b) — a plain integral over the simplex.
- Side info: states y split the tape into subsequences; the state-conditional
  universal portfolio competes with the best STATE-constant-rebalanced
  portfolio; regret (d/2n)log(n+1) + (k/n)log2 with d = k(m-1).

## Usage

```
# research: 400 shared block-bootstrap paths (block 21, seed 0 — same paths as
# cppi_backtest.py item 14), universal vs erc/hrp/cppi_m3/vincent_ls
python universal_portfolio.py paths --paths 400 --save
# add the HMM side-information variant (universal_side book)
python universal_portfolio.py paths --paths 400 --side --save

# personal book: daily universal weights for the 9 book names, Cover sec.10
# cost gate, and the sizing plan vs holdings
python universal_portfolio.py book --save --sizing [--tickers ...] [--cost 0.001]

# sizing plan only (reads saved book weights + holdings)
python universal_portfolio.py sizing --save [--value $]

# engine validation (exact vs MC on synthetic 2-asset data)
python universal_portfolio.py validate --samples 100000
```

Standard `cli_common` flags do not apply (standalone research tool).

## Outputs

| Output | Producer | Family |
|---|---|---|
| `universal_paths.parquet` | `universal_portfolio.py paths` | Other |
| `universal_book_weights.parquet` | `universal_portfolio.py book` | Other |
| `universal_sizing_plan.parquet` | `universal_portfolio.py book --sizing` / `sizing` | Other |

- `universal_paths.parquet`: book × (terminal, maxdd, underwater_days) over 400
  bootstrap paths — same stats schema as `cppi_paths.parquet` (item 14);
  `--side` adds the `universal_side` book.
- `universal_book_weights.parquet`: date × ticker × weight (cost-gated) for the
  personal book. Derived panel — the script only READS `daily_prices/`.
- `universal_sizing_plan.parquet`: ticker × current/target weight/notional ×
  delta shares at last close × action (gate-honoring).

## Measured results (see docs/RESEARCH_INTEGRATION_PLAN.md item 23)

- Research bar (same as item 14) **PASS**: universal median terminal 1.96 vs
  ERC 1.86, maxDD 20.9% vs 21.5% — the no-lookahead baseline beats the
  risk-parity claim on both axes. Side-info (HMM) is a wash on the index tape
  (1.94, 0.99×) — the best state-CRP ≈ best CRP there. Vince LS still owns
  the leverage claim (4.37×, 30% DD).
- Book: gated universal 2.85× vs equal-weight 2.58× on the common window
  (2,799 days); gate fires on only 24 days and improves wealth (the gate
  filters rebalance noise, not just costs). Sizing plan live: trims
  BAYRY/CAG/HMC/KHC/T, adds HPQ/PFE/SMCI.

## Related programs

- `cppi_backtest.py` — item 14 (CPPI vs ERC vs HRP vs Vince LS; same paths)
- `kelly.py` — Vince leverage space / log-optimal RS (the CRP-oracle family)
- `hmm_regime_detection.py` — the HMM whose features `RegimeStates` reuses for
  the side-information states
- `build_bogle_funds.py` — TMI/BPI, the 2-asset substrate for `paths`
- docs/cover_universal_portfolio.md — the research brief (theory + exact tables)

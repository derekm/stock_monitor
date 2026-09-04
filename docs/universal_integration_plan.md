# Universal Portfolio — Ad-Hoc Integration & Personal Portfolio Rebalance Plan

**Date:** 2026-09-04. **Inputs:** `universal_sizing_plan.parquet` (Cover §10
gated targets, 0.10% cost), `portfolio_holdings.parquet`, `preferred_metrics`,
`signal_aggregator_scores`, `implied_r_screen`, `tail_index_robust`,
`fragility_veto`, `momentum_metrics`, `ride_book`, `buy_candidates`,
`vol_targets`, `rebalance_calendar`.

**Method:** the universal portfolio is the regret-optimal **no-lookahead**
allocation baseline — it is NOT alpha and it has no risk layer. The integration
rule is therefore: **universal sets the direction; gates and signals veto the
size.** Veto priority: fragility veto > ride score + momentum (tilt confirm/deny)
> vol-target sizing > implied-r (value confirm) > composite signal > universal.

## 1. Integrated book view (2026-09-02 close)

| ticker | cur % | univ target % | Δ pp | veto | α | mom 12-1 | implied r | ride | composite | integrated action |
|---|---|---|---|---|---|---|---|---|---|---|
| BAYRY | 13.8 | 9.1 | **−4.7** | — | 3.56 | **+62.6%** | — | 0.41 | 0.612 | **trim hard** (winner, +30.6% unreal) |
| CAG | 11.9 | 9.3 | −2.6 | — | 3.76 | −15.6% | — | 0.40 | 0.504 | trim |
| HMC | 11.5 | 10.1 | −1.4 | — | 3.99 | −8.8% | — | 0.43 | 0.655 | trim |
| HPQ | 9.4 | 11.8 | **+2.4** | — | 3.78 | +4.5% | **16.7%** | 0.42 | **0.840** | **add** (cleanest signal stack) |
| KHC | 10.6 | 9.1 | −1.5 | — | 2.67 | +0.5% | — | 0.49 | 0.702 | trim |
| MOS | 9.7 | 9.7 | +0.1 | — | 4.02 | −25.9% | 3.2% | 0.42 | 0.541 | hold (at target) |
| PFE | 9.2 | 10.4 | +1.2 | — | 5.69 | +11.4% | 6.5% | 0.40 | 0.697 | add small |
| **SMCI** | 12.3 | 19.5 | **+7.2** | **⚠️ TRUE** | 3.36 | **−24.4%** | 9.5% | **0.34 (worst)** | 0.717 | **TRIM — veto overrides universal add** |
| T | 11.8 | 11.1 | −0.7 | — | 4.52 | −16.8% | **16.5%** | 0.40 | 0.580 | hold (universal says trim; implied-r says keep) |

**Headline:** the universal portfolio's single biggest move — overweight SMCI to
19.5% — is **rejected by five independent layers** (fragility veto, ride score
worst-on-book, negative 12-1 momentum, vol-target sizing wanting −0.64 sh, and
negative price momentum). Universal is hindsight-momentum by construction; SMCI
is exactly the case where the no-risk baseline must lose to the risk layer.
Every other universal move survives the veto screen; HPQ is the strongest
conviction add (top composite 0.840 + top implied-r 16.7% + positive momentum).

## 2. Rebalance schedule

Follow the existing month-end calendar (`rebalance_calendar.parquet`: last
trading day, stress-conditional turnover band 0.62 when p_stress > 0.5), with
two integration rules:

1. **Cover §10 gate first:** execute a name only when |Δ weight| > cost drag
   (0.10% → 0.5 pp practical band, since a 1-share BAYRY trim is −0.33 pp).
   The 2,799-day backtest fired the gate on 24 days (~quarterly).
2. **Veto overrides:** any name with `veto_flag = TRUE` is sized DOWN from its
   **current** weight, never toward the universal target; α < 2.0 or ride < 0.35
   triggers the same downgrade.
3. **Stagger for liquidity:** ADR/mid-cap names (BAYRY, HMC, KHC, MOS) trade at
   the month-end rebalance; large liquid names (HPQ, T, PFE, SMCI, CAG) may be
   adjusted within the month when the added spread is > 50 bps vs month-end.

Recommended cadence: **month-end full review; intra-month only for veto
breaches (SMCI) and limit-order fills.**

## 3. Aggressive profit-taking limit sell ladder

**Brokerage note (Robinhood/retail brokers): fractional orders are MARKET-ONLY
— limit orders require whole shares.** So the ladder below is conceptual: the
only *resting limit* orders you can place are whole-share sells on positions
large enough (BAYRY). All fractional trims must be executed as market orders
at the moment you choose — you give up the resting-price ladder, not the
direction. The SMCI front-load was likewise executed as ONE market sell
(0.4 sh @ $39.72, filled 2026-09-04 — above both rungs, recorded trade 17).

Executable version for the current book (prices/deltas recalc'd 2026-09-04):

| ticker | hold (sh) | trim | executable order |
|---|---|---|---|
| BAYRY | 3.00 | −1.03 | whole-share LIMIT sell 1.0 sh (e.g. ≥ $15.00) — the only resting-sell |
| CAG | 2.31 | −0.55 | fractional MARKET sell ~0.5 sh |
| KHC | 1.31 | −0.24 | fractional MARKET sell ~0.25 sh |
| T | 1.47 | −0.16 | skip (≈0) or fractional MARKET sell ~0.15 sh |
| HMC | 1.10 | −0.14 | fractional MARKET sell ~0.15 sh |
| MOS | 1.30 | −0.13 | fractional MARKET sell ~0.13 sh |
| HPQ | 1.00 | +0.14 | fractional MARKET buy ~$4 (small, no limit available) |
| PFE | 1.03 | +0.06 | skip (≈0) |
| SMCI | 0.66 | (add vetoed) | **done — 0.4 already sold, hold the rest** |

Rules: limit fill = trim executed at the rung; if the stock gaps through, the next rung
up catches the remainder (aggressive-by-design, no trailing stop needed).

| ticker | last | trim (sh) | rung 1 (⅓) | rung 2 (⅓) | rung 3 (⅓) | note |
|---|---|---|---|---|---|---|
| **SMCI** | 36.50 | **−0.40** | **37.60 (+3%) 0.20 sh** | **39.42 (+8%) 0.20 sh** | — | veto front-load; never buy the universal add |
| BAYRY | 14.37 | −1.02 | 15.09 (+5%) 0.35 sh | 15.81 (+10%) 0.35 sh | 16.53 (+15%) 0.32 sh | biggest trim, +30.6% unreal |
| CAG | 16.27 | −0.50 | 17.08 (+5%) 0.17 sh | 17.90 (+10%) 0.17 sh | 18.71 (+15%) 0.16 sh | |
| HMC | 32.60 | −0.13 | 34.23 (+5%) 0.05 sh | 35.86 (+10%) 0.04 sh | 37.49 (+15%) 0.04 sh | |
| KHC | 25.57 | −0.19 | 26.85 (+5%) 0.07 sh | 28.13 (+10%) 0.06 sh | 29.41 (+15%) 0.06 sh | |
| T | 25.15 | −0.09 | 26.41 (+5%) 0.05 sh | 27.67 (+10%) 0.04 sh | — | implied-r 16.5% → smallest trim |

BUY limit pairs (for the adds, at month-end or on ≥3% pullback): HPQ ≤ 28.60
(−3%), PFE ≤ 26.95 (−3%). MOS holds at target — no order.

## 4. Expected effect

Trim ~3.9 pp of BAYRY and ~0.9 pp average from the five trim names, recycle
~$45–55 of the book ($31,345 total) into HPQ (+0.26 sh @ ≤28.60) and PFE
(+0.13 sh @ ≤26.95); ALWAYS skip the SMCI add. Target book after fills:
HPQ ≈ 11.8%, PFE ≈ 10.4%, SMCI ≈ 11% (down from 12.3%), others near target.

**Do not:** size SMCI up (five-layer veto). Chase the universal add beyond the
limit pair prices. Rebalance on days when the calendar says stress
(turnover band 0.62 applies — halve all execution sizes). Treat this as a
mechanical target + gate overlay, not a view: anything below the veto layer is
baseline, not opinion.

**Output:** this plan is generated from the live tables; re-run
`universal_portfolio.py book --save --sizing` after every month-end close to
refresh the ladder.

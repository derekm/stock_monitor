# hidden_optionality_audit.py — which point estimates ride on noise?

## Description
The American-options lesson applied to decision-making: every quantity a
system treats as deterministic while it is actually stochastic contains
unpriced convexity (El Hassan, Maddah & Taleb 2026). This audit perturbs each
decision driver of `buy_candidates.py` by its OWN estimation error and
measures the decision flip rate — the probability a BUY/ACCUMULATE/WATCH/AVOID
verdict is riding on noise.

## Why it exists (rationale)
The American-options paper shows the early-exercise feature is optionality on
the path of the rate differential — invisible to models that fix r₁−r₂. Our
stack fixes several "r₁−r₂" values: the HMM regime label (used as a hard
stress verdict), momentum_score, factor_composite, and the aggregate composite.
The paper's method (§I-A): stochasticize one input at a time and measure the
convexity bias π = E[f(ã)] − f(E[ã]) — "a bad ruler might not give us the
precise height of a growing child, but will inform us whether the child is
growing."

## Usage
```
python hidden_optionality_audit.py [--n-perturb 200] [--seed 7]
```
Requires the buy_candidates inputs (preferred/momentum/factor/risk/aggregate
CSVs) + hmm_regime_states.csv — i.e. run the analytics first.

## Outputs (see SCHEMAS → Taleb / fat tails family)
- `hidden_optionality.csv` — per driver: perturb_scale (its estimation error),
  flip_rate (decision flips / trials × tickers), mean_score_move, n.

## Findings (2026-08 run)
The HMM regime label is the single most fragile point estimate: **28.4% of
decisions flip** when the stress verdict changes (perturb scale 0.10).
Momentum/factor/composite each flip ~26% at their own estimation-error scales.

## Related
buy_candidates.py (the decisions being audited), regime_forecast.py
(hmm_regime_states), the Forecasting-Paradox upgrades in forecast_granite.py
(--epistemic-error, forecast_nu). Registered as the `taleb_optionality` daily
job (after aggregate + preferred).

# ergodicity_ruin.py — The ergodicity critique

## Description
Computes the TIME-average quantities that actually determine whether a
compounding portfolio survives: arithmetic vs geometric return gap (variance
drag), ruin probability over 1/5/10y horizons from block-bootstrap paths,
time-to-double vs time-to-ruin, and terminal-wealth percentiles.

## Why it exists (rationale)
Sharpe ratios and ensemble averages are NOT ergodic for fat-tailed payoffs —
the time average differs from the ensemble average, and ruin is absorbing
(you cannot recover from -100% with +100%). Two strategies with identical
Sharpe can have wildly different ruin probabilities.

## Usage
```
python ergodicity_ruin.py [--years 1 5 10] [--paths 400] [--tickers A,B]
```

## Outputs (see SCHEMAS → `taleb` family)
- `ergodicity_ruin.csv` — per-ticker arith/geom annual return, variance drag,
  ruin prob (P(terminal < 0.5)) per horizon, P(99% drawdown), days-to-double,
  days-to-5%-drawdown, double/ruin ratio.
- `portfolio_ergodic.csv` — same for the equal-weight portfolio + terminal
  wealth p5/p50/p95 per horizon.

## Related
tail_index.py (the fat tails that make the ergodicity gap large),
barbell_check.py. Registered as the `taleb_ergodic` daily job (after tail).

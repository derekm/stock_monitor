# gap_risk.py — Overnight gap exposure

## Description
Measures the risk that arrives in OVERNIGHT GAPS (open vs prev close), which
close-to-close backtests cannot see. Gap share of variance > 1 means a name's
risk is overnight-dominated — these names gap through stops.

## Why it exists (rationale)
Most catastrophic equity losses happen overnight, not intraday. A name can
look calm on close-to-close returns while its gaps carry all the tail risk.
This is the first place to look when a "low-vol" name blows up.

## Usage
```
python gap_risk.py [--top-events 40]
```

## Outputs (see SCHEMAS → `taleb` family)
- `gap_risk.csv` — per-ticker mean gap, gap sd, gap share of total variance,
  P(|gap|>3%), P(|gap|>5%), max/min gap %.
- `gap_events.csv` — top-N worst single-day gaps with dates.

## Related
tail_index.py, fragility_screen.py (consumes gap_share_of_var),
barbell_check.py (gap share defines the convex bucket). Registered as the
`taleb_gap` daily job. Runs before `taleb_fragility`.

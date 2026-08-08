# fragility_screen.py — Per-name fragility index

## Description
Combines eight fragility inputs (leverage, asset coverage, interest coverage,
IV skew, illiquidity, gap share, tail fatness, kurtosis) into a composite
fragility score and percentile, with a FRAGILE veto flag (top 10%).

## Why it exists (rationale)
Cheap is not enough: a name can be cheap AND fragile, and fragile names get
destroyed in the tails regardless of valuation. Fragility = sensitivity to
volatility and vol-of-vol. The flag is a VETO for inclusion — cheap + fragile
= skip. The market's own fragility gauge (IV skew) is a component.

## Usage
```
python fragility_screen.py
```

## Outputs (see SCHEMAS → `taleb` family)
- `fragility_screen.csv` — per-ticker component percentiles, composite score,
  fragility percentile, fragile_flag (True = top 10% = veto).

## Consumed by
- buy_candidates.py — fragile names get `fragile_veto` (−0.30 score) and
  skew-steep names get `skew_steepening` (−0.15).
- shadow_book.py — a held name flagged fragile triggers the proactive
  `fragile@<date>(<names>)` kill switch (exit before the loss).
- barbell_check.py — average fragility scales the convexity allocation.

## Related
tail_index.py, gap_risk.py (inputs), options_skew.csv (input). Registered as
the `taleb_fragility` daily job (after taleb_tail + taleb_gap).

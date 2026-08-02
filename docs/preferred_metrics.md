# preferred_metrics.py

Automated **preferred metrics** for screening, sizing, and inclusion decisions.

## Metric stack

### Buffett-style quality
| Metric | Prefer |
|--------|--------|
| ROE | ≥ **15%** |
| ROIC | ≥ **15%** |
| Debt/Equity | ≤ **1.0** (ideal ≤ 0.5) |
| Interest coverage | higher better |
| Earnings stability | 0–1 score (predictability) |

### Value trifecta (prior threads)
| Metric | Prefer |
|--------|--------|
| EV/EBITDA | ≤ **9** |
| P/B | ≤ **1.5** |
| MktCap/Assets | ≤ **0.5** |

### Sizing overlays
- Composite score → suggested max weight bands (3–12%)
- **SMCI hard cap 5%** (vol-target aware when `vol_targets.csv` present)
- Actions: `prefer_add` / `hold_or_add` / `hold` / `reduce_or_avoid`

## Decision labels

| Label | Meaning |
|-------|---------|
| **INCLUDE_CORE** | Buffett pass **and** trifecta pass |
| **INCLUDE_VALUE** | Trifecta pass |
| **INCLUDE_QUALITY** | Buffett pass |
| **SATELLITE** | Solid composite, not both screens |
| **WATCH** / **AVOID** | Weak composite |

```bash
python preferred_metrics.py --seed-quality --save
python preferred_metrics.py --decision INCLUDE_VALUE
python check_alerts.py --dry-run   # VALUE_TRIFECTA + BUFFETT_QUALITY rules
```

Outputs: `preferred_metrics.csv`, `preferred_screen_hits.csv`  
Quality fields are seeded as **approx** until live fundamentals replace them (`quality_source=seed_approx_buffett`).

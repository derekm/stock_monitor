# dupont_analysis.py

DuPont decomposition of ROE:


$$
\mathrm{ROE} \approx \underbrace{\mathrm{PM} \times \mathrm{Asset\ Turnover}}_{\text{pm x at}} \times \underbrace{(1 + D/E)}_{\text{equity multiplier}}
$$


**Buffett preference:** high ROE from **operations** (`high_ops`: EM ≤ 1.5), not from leverage (`leverage_driven`).

```bash
python dupont_analysis.py --save
```

Output: `dupont_analysis.csv`

## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/update_fundamentals.md](update_fundamentals.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

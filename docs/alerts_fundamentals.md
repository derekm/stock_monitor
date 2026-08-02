# Fundamental / trifecta alerts

## Rule types

| Type | Purpose |
|------|---------|
| `trifecta` / `fundamentals_screen` | Multi-metric screen via `conditions` |
| `metric_below` / `metric_above` | Single fundamental metric |

## Conditions syntax

```
ev_ebitda<=9;pb_ratio<=1.5;mktcap_to_assets<=0.5
```

- `match_mode=all` → AND (strict trifecta)
- `match_mode=any` → OR (soft screen)

## Built-in rules

- **VALUE_TRIFECTA** (high): EV/EBITDA≤9 AND P/B≤1.5 AND MktCap/Assets≤0.5
- **VALUE_ANY_TWO** (medium): any leg of the trifecta
- **LOW_EV_EBITDA** (medium): EV/EBITDA≤9

```bash
python manage_alerts.py add --rule-id MY_SCREEN --ticker '*' \
  --type fundamentals_screen --conditions 'ev_ebitda<=8;pb_ratio<=1.2' \
  --match-mode all --priority high --notes 'tight value'

python check_alerts.py --dry-run
python check_alerts.py --priority high
```

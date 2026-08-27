# check_alerts.py

Evaluate configured alert rules against current price history.

## Why it exists (rationale)

Screens and weights are static; prices move. This script runs the rule set in
`alerts_config.parquet` (managed by `manage_alerts.py`) against live
`daily_prices/` and logs hits — the active monitoring layer that tells you
when a threshold (price, fundamentals, or screen change) actually trips.

## Usage

```bash
python check_alerts.py                  # run all enabled rules, print & log
python check_alerts.py --dry-run        # print only, do not write log
python check_alerts.py --priority high  # only high-priority rules
python check_alerts.py --ticker CF      # only rules that apply to CF (or *)
python check_alerts.py --list-rules     # show the current rule set
```

Flags: `--dry-run`, `--priority` (high/medium/low), `--ticker`, `--list-rules`.

## Outputs

- `alerts_log.parquet` — appended alert events (unless `--dry-run`)
- Reads `alerts_config.parquet` (rule set) and `daily_prices/`

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [manage_alerts.md](manage_alerts.md) — add/enable/disable rules
- [update_prices.md](update_prices.md) — freshens the price input
- [preferred_metrics.md](preferred_metrics.md) — fundamentals-based rules

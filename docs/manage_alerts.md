# manage_alerts.py

Add, enable/disable, or list the alert rules stored in `alerts_config.parquet`.

## Why it exists (rationale)

`check_alerts.py` evaluates rules; this edits them. It is the write side of the
alerting subsystem — the way you actually add a price/fundamental rule, flip it
on/off, or audit the current set. The dashboard's Alerts tab stages JSON that
`apply-json` consumes here.

## Usage

```bash
python manage_alerts.py list [--enabled-only]
python manage_alerts.py enable RULE_ID
python manage_alerts.py disable RULE_ID
python manage_alerts.py add --rule-id MY_RULE --ticker CF --type price_above \
       --param1 140 --priority high --notes "..."
```

Sub-commands: `list`, `enable`, `disable`, `add`. `add` flags: `--rule-id`,
`--ticker`, `--type`, `--param1`, `--param2`, `--priority`, `--notes`.

## Outputs

- `alerts_config.parquet` — the rule set (mutated in place)

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [check_alerts.md](check_alerts.md) — evaluates these rules
- [analytics_service.md](analytics_service.md) — `POST /run/alerts`

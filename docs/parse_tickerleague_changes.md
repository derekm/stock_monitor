# parse_tickerleague_changes.py

Extract the FULL S&P 500 additions & removals history from tickerleague.com —
the primary data source behind `parse_sp500_changes`.

## Why it exists (rationale)

tickerleague.com embeds the complete ADD/REMOVE dataset (back to the 1950s,
31 client-paginated pages) as a JS-stringified JSON array in a `<script>` tag;
the visible HTML table only shows 50 rows. A strict `json.loads` fails on a few
rows with inner quoted phrases, so this uses a tolerant field scanner to recover
the whole history. `parse_sp500_changes` calls the equivalent logic and falls
back to Wikipedia; this script is the standalone scraper.

## Usage

```bash
python parse_tickerleague_changes.py
```

Flags: none. Network fetch from tickerleague.com.

## Outputs

- `sp500_changes_tickerleague.parquet` — full event history

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [parse_sp500_changes.md](parse_sp500_changes.md) — wraps this (with Wikipedia fallback)
- [parse_sp500.md](parse_sp500.md)
- [sp_universe_tracking.md](sp_universe_tracking.md)

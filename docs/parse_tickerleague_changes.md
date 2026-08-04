# parse_tickerleague_changes.py

Extract the FULL S&P 500 additions & removals history from tickerleague.com (the data is embedded as a JS-stringified JSON array in a <script> tag; the site paginates client-side over 31 pages back to the 1950s).

## Why it exists (rationale)

Extracts the full S&P 500 additions/removals history from tickerleague.com into an event log used by `sp_index_methodology`.

## Usage

```bash
python parse_tickerleague_changes.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/parse_sp500_changes.md](parse_sp500_changes.md)
- [docs/sp_index_methodology.md](sp_index_methodology.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)

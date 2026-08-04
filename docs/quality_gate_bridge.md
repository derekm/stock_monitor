# quality_gate_bridge.py

Canonical dual-screen quality/value gate for stock_monitor — a bridge to the
`stockmagic` analytics library (the source of truth for the Buffett + trifecta +
leverage gates).

## Why it exists (rationale)

`dual_screen_analysis.py` historically hard-coded its own Buffett/trifecta
thresholds, which drifted from the canonical gate in `stockmagic/src/analytics/
quality_value.py`. This bridge makes **one** gate authoritative and adds PIT
correctness: a screen "as of" a date only sees fundamentals reported on/before
that date. It queries `fundamentals.parquet` / `monitored_stocks.parquet` via
DuckDB.

## Usage

```bash
python quality_gate_bridge.py            # prints gate evaluation for current tickers
```

Flags: minimal (reads the parquet sources via DuckDB; see source for any
sub-commands). Primarily an importable module for other scripts.

## Outputs

None written directly (returns gate results; consumed by `dual_screen_analysis`
and the screen layer). See [SCHEMAS.md](SCHEMAS.md).

## Related programs

- [dual_screen_analysis.md](dual_screen_analysis.md) — uses this bridge
- [inclusion_criteria.md](inclusion_criteria.md) — the stock_monitor gate it canonicalizes
- [preferred_metrics.md](preferred_metrics.md)
- (external) `stockmagic/src/analytics/quality_value.py` — the canonical gate

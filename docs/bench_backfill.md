# bench_backfill.py

Device-throughput benchmark: CPU vs MX550 for tiny-model (TTM-class, ~1M param)
rolling-window time-series training. Isolates whether the MX550 gives a speedup
for the small-batch backfill workload, using a generic small model so it does
not depend on `tsfm_public` / the Granite checkpoint.

## Why it exists (rationale)

Answers a hardware question for the backfill design: is the MX550 worth using,
or is CPU fine? It times a small TCN-ish model over 512→96 windows for a sample
of 20 large-cap tickers on each device.

## Usage

```bash
python bench_backfill.py
```

Flags: none (hard-coded sample + context/horizon 512/96). Prints per-device
timing to stdout.

## Outputs

None written to disk. Prints timing comparisons to stdout.

## Related programs

- `_cuda_bench.py` / `_par_bench.py` / `_feed_test.py` — related backfill
  throughput probes (developer scratch)
- [ttm_backfill.md](ttm_backfill.md) / [granite_backfill.md](granite_backfill.md)
  — the backfill it benchmarks

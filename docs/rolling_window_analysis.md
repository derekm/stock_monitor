# rolling_window_analysis.py

Rolling vol, beta, Sharpe, max-drawdown, and dual-screen stability — now with GPU-accelerated cumsum and resumable checkpoints.

## Why it exists (rationale)

Single-point risk stats hide regime dependence. This rolls vol/beta/Sharpe/DD (and dual-screen pass stability) over a trailing window per ticker, so you can see whether a name's risk profile is stable or deteriorating — input to sizing and the buy decision.

## Usage

```bash
python rolling_window_analysis.py --universe all --window 63 --save
python rolling_window_analysis.py --universe portfolio --window 252 --save
```

Flags:
- `--universe`: `all`, `portfolio`, `index_member`, or comma-separated tickers (default: `all`)
- `--window`: Rolling window in days (default: 63)
- `--save`: Write output parquet
- `--gpu`: Force GPU device (`cuda`, `directml`, `cpu`)

Reads `daily_prices.parquet`, `monitored_stocks.parquet`, `portfolio_holdings.parquet`, `preferred_metrics.parquet`.

## Key Features (2026-08)

- **GPU acceleration**: Uses `tensor_ops.rolling_cumsum_2d` for vectorized cumsum on CUDA or DirectML (Intel Xe). Falls back to CPU NumPy automatically.
- **Resumable checkpoints**: Uses `resumable_job.JobCheckpoint` — skips tickers already computed (prints `=== Rolling 63d SKIP N tickers already complete ===`).
- **Universe from daily_prices**: No longer uses `monitored_stocks.parquet` as universe source.
- **Output**: `rolling_window_metrics.parquet` with columns: `ticker`, `as_of_date`, `window`, `vol`, `beta`, `sharpe`, `max_dd`, `dual_screen_stable`, `data_provenance`.

## Performance

| Universe | Tickers | Window | CUDA (MX550) | DirectML (Xe) | CPU |
|----------|---------|--------|--------------|---------------|-----|
| all | ~10k | 63 | ~0.12s | ~0.20s | ~0.03s |
| all | ~10k | 252 | ~0.16s | ~0.33s | ~0.44s |

## Outputs

- `rolling_window_metrics.parquet` — per-ticker rolling risk metrics (Schema family: `summary_metrics` — see [SCHEMAS.md](SCHEMAS.md).)
- Checkpoint: `backfill_checkpoints/rolling_window_analysis_daily_prices.json`

## Related programs

- [risk_enrich.md](risk_enrich.md) / [risk_metrics_ext.md](risk_metrics_ext.md)
- [preferred_metrics.md](preferred_metrics.md)
- [momentum_analytics.md](momentum_analytics.md)
- [tensor_ops.py](tensor_ops.py) — unified GPU/CPU rolling operations
- [resumable_job.md](resumable_job.md) — checkpoint framework
# resumable_job.py
Checkpoint mechanism for full-universe full-history jobs.

## Why it exists (rationale)

Full-universe, full-history jobs (e.g., reprocessing 10,000 tickers × 15 years) can take hours or days and fail midway. Without checkpointing, every failure means starting over. This framework provides per-ticker progress tracking with file-based checkpoints in `backfill_checkpoints/`, automatic detection of universe changes (new/removed tickers) and source data changes (new backfill runs), and forced full-reload support. Jobs resume from the last completed ticker, skipping work that's already done.

## Usage

```python
from resumable_job import run_resumable_job

def my_process_fn(ticker: str, last_date: date | None) -> date:
    # Process ticker, return last date processed
    return date.today()

stats = run_resumable_job(
    job_name="my_backfill_job",
    process_ticker_fn=my_process_fn,
    universe_source="daily_prices",
    force_reload=False,
)
```

CLI (example mode):

```bash
python resumable_job.py --job rolling_window_analysis       # run with checkpoint
python resumable_job.py --job rolling_window_analysis --full-reload  # force restart
```

## Outputs

- `backfill_checkpoints/<job_name>_checkpoint.json` — per-ticker progress, universe hash, data hash, timestamps
- `backfill_checkpoints/<job_name>_checkpoint.lock` — file lock for atomic updates

Locks: `msvcrt` on Windows, `fcntl` elsewhere. Not wired into `run_daily_automation.py` yet — jobs still run full-pass unless they call this module themselves.

## Related programs

- `run_daily_automation.py` — orchestrator that runs daily jobs (candidate for resumable integration)
- `backfill_edgar.py` — example of a long-running job that benefits from checkpointing
- `backfill_checkpoints/` — directory where checkpoint files are stored
#!/usr/bin/env python3
"""
Resumable Job Framework — Checkpoint mechanism for full-universe full-history jobs.

Key features:
- Checkpoint files in backfill_checkpoints/ directory
- Detects new backfill runs that invalidate checkpoints
- Per-ticker progress tracking with last processed date
- Supports forced full-reload via --full-reload flag
- Schema versioning with migration support
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import json
import hashlib
import os
import sys

CHECKPOINT_DIR = Path(__file__).parent / "backfill_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Current schema version - increment when checkpoint format changes
CURRENT_SCHEMA_VERSION = 2


def _migrate_checkpoint(state: dict) -> dict:
    """Migrate old checkpoint format to current schema version."""
    if not state:
        return state

    version = state.get("schema_version", 1)
    if version >= CURRENT_SCHEMA_VERSION:
        return state

    # Migration from v1 to v2: add migration metadata
    if version == 1:
        state = state.copy()
        state["schema_version"] = CURRENT_SCHEMA_VERSION
        state["migrated_at"] = datetime.now().isoformat()
        state["migrated_from"] = 1
        # Ensure required v2 fields exist
        if "tickers" not in state:
            state["tickers"] = {}
        if "failed_tickers" not in state:
            state["failed_tickers"] = []
        if "completed_tickers" not in state:
            state["completed_tickers"] = len([
                t for t, info in state.get("tickers", {}).items()
                if info.get("status") == "complete"
            ])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class JobCheckpoint:
    """
    Manages checkpoint state for a daily automation job.

    Checkpoint file structure (schema v2):
    {
        "job_name": "rolling_window_analysis",
        "schema_version": 2,
        "universe_hash": "sha256 of sorted ticker list",  # Detects new/removed tickers
        "data_hash": "sha256 of data source files",       # Detects data changes
        "last_run": "2026-08-16T10:00:00",
        "tickers": {
            "AAPL": {"last_date": "2026-07-31", "status": "complete"},
            "MSFT": {"last_date": "2026-07-31", "status": "complete"},
            ...
        },
        "total_tickers": 9954,
        "completed_tickers": 9954,
        "failed_tickers": [],
        "migrated_at": "2026-08-17T10:00:00",  # v2+
        "migrated_from": 1                     # v2+
    }
    """

    def __init__(self, job_name: str, universe_source: str = "daily_prices"):
        self.job_name = job_name
        self.universe_source = universe_source
        self.checkpoint_file = CHECKPOINT_DIR / f"{job_name}_checkpoint.json"
        self.lock_file = CHECKPOINT_DIR / f"{job_name}_checkpoint.lock"
        self._state = None
        self._lock_fd = None

    def _compute_universe_hash(self, tickers: list[str]) -> str:
        """Hash of sorted ticker list to detect universe changes."""
        ticker_str = ",".join(sorted(tickers))
        return hashlib.sha256(ticker_str.encode()).hexdigest()[:16]

    def _compute_data_hash(self) -> str:
        """Hash of source data files to detect new backfill runs."""
        import hashlib

        files_to_hash = []
        if self.universe_source == "daily_prices":
            files_to_hash = ["daily_prices/"]
        elif self.universe_source == "fundamentals":
            files_to_hash = ["fundamentals.parquet"]
        elif self.universe_source == "both":
            files_to_hash = ["daily_prices/", "fundamentals.parquet"]

        combined_hash = hashlib.sha256()
        for f in files_to_hash:
            fp = Path(__file__).parent / f
            if fp.exists():
                stat = fp.stat()
                # Hash based on size + mtime (fast, detects changes)
                file_info = f"{f}:{stat.st_size}:{stat.st_mtime_ns}"
                combined_hash.update(file_info.encode())

        return combined_hash.hexdigest()[:16]

    def _acquire_lock(self):
        """Acquire file lock. msvcrt on Windows, fcntl elsewhere."""
        self._lock_fd = open(self.lock_file, "a+")
        if sys.platform == "win32":
            import msvcrt
            self._lock_fd.seek(0)
            try:
                msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_LOCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)

    def _release_lock(self):
        """Release file lock."""
        if self._lock_fd:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self._lock_fd.seek(0)
                    msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._lock_fd.close()
            self._lock_fd = None

    def load(self) -> dict | None:
        """Load checkpoint state, returning None if invalid or missing."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file) as f:
                state = json.load(f)

            # Migrate if needed
            state = _migrate_checkpoint(state)

            # Validate schema
            if state.get("schema_version") != CURRENT_SCHEMA_VERSION:
                return None
            if state.get("job_name") != self.job_name:
                return None

            self._state = state
            return state
        except Exception:
            return None

    def is_valid(self, current_tickers: list[str], force_reload: bool = False) -> bool:
        """
        Check if checkpoint is valid for current run.

        Invalid if:
        - force_reload=True
        - No checkpoint exists
        - Universe changed (tickers added/removed)
        - Source data changed (new backfill detected)
        """
        if force_reload:
            return False

        state = self.load()
        if state is None:
            return False

        # Check universe hash
        current_hash = self._compute_universe_hash(current_tickers)
        if state.get("universe_hash") != current_hash:
            print(f"  [{self.job_name}] Universe changed (hash mismatch) - full reload required")
            return False

        # Check data hash
        current_data_hash = self._compute_data_hash()
        if state.get("data_hash") != current_data_hash:
            print(f"  [{self.job_name}] Source data changed (new backfill detected) - full reload required")
            return False

        return True

    def get_completed_tickers(self) -> set[str]:
        """Get set of tickers already processed in current checkpoint."""
        if self._state is None:
            self.load()
        if self._state is None:
            return set()

        completed = set()
        for ticker, info in self._state.get("tickers", {}).items():
            if info.get("status") == "complete":
                completed.add(ticker)
        return completed

    def get_ticker_last_date(self, ticker: str) -> date | None:
        """Get last processed date for a ticker."""
        if self._state is None:
            self.load()
        if self._state is None:
            return None

        info = self._state.get("tickers", {}).get(ticker)
        if info and info.get("last_date"):
            return datetime.fromisoformat(info["last_date"]).date()
        return None

    def mark_ticker_started(self, ticker: str):
        """Mark a ticker as in-progress."""
        self._acquire_lock()
        try:
            if self._state is None:
                self.load()
            if self._state is None:
                self._init_state(current_tickers=[])

            self._state["tickers"][ticker] = {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "last_date": None,
            }
            self._save()
        finally:
            self._release_lock()

    def mark_all_complete(self, tickers: list[str], last_date: date):
        """Mark a batch of tickers complete (one lock)."""
        self._acquire_lock()
        try:
            if self._state is None:
                self._init_state(tickers)
            iso = last_date.isoformat()
            now = datetime.now().isoformat()
            for t in tickers:
                self._state["tickers"][t] = {"status": "complete", "completed_at": now, "last_date": iso}
            self._state["completed_tickers"] = len(self.get_completed_tickers())
            self._save()
        finally:
            self._release_lock()

    def mark_ticker_complete(self, ticker: str, last_date: date):
        """Mark a ticker as complete with its last processed date."""
        self._acquire_lock()
        try:
            if self._state is None:
                self.load()
            if self._state is None:
                return

            self._state["tickers"][ticker] = {
                "status": "complete",
                "completed_at": datetime.now().isoformat(),
                "last_date": last_date.isoformat(),
            }
            self._state["completed_tickers"] = len(self.get_completed_tickers()) + 1
            self._save()
        finally:
            self._release_lock()

    def mark_ticker_failed(self, ticker: str, error: str):
        """Mark a ticker as failed."""
        self._acquire_lock()
        try:
            if self._state is None:
                self.load()
            if self._state is None:
                return

            self._state["tickers"][ticker] = {
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": error,
            }
            self._state.setdefault("failed_tickers", []).append(ticker)
            self._save()
        finally:
            self._release_lock()

    def _init_state(self, current_tickers: list[str]):
        """Initialize new checkpoint state."""
        self._state = {
            "job_name": self.job_name,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "universe_hash": self._compute_universe_hash(current_tickers),
            "data_hash": self._compute_data_hash(),
            "created_at": datetime.now().isoformat(),
            "last_run": datetime.now().isoformat(),
            "tickers": {},
            "total_tickers": len(current_tickers),
            "completed_tickers": 0,
            "failed_tickers": [],
        }

    def _save(self):
        """Save checkpoint state to disk."""
        if self._state:
            self._state["last_run"] = datetime.now().isoformat()
            with open(self.checkpoint_file, "w") as f:
                json.dump(self._state, f, indent=2)

    def initialize(self, current_tickers: list[str]):
        """Initialize checkpoint for a new run."""
        self._acquire_lock()
        try:
            self._init_state(current_tickers)
            self._save()
        finally:
            self._release_lock()

    def finalize(self):
        """Mark job as fully complete."""
        self._acquire_lock()
        try:
            if self._state:
                self._state["completed_at"] = datetime.now().isoformat()
                self._save()
        finally:
            self._release_lock()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_tickers(source: str = "daily_prices") -> list[str]:
    """Get current universe of tickers from data source."""
    from pathlib import Path
    DATA_DIR = Path(__file__).parent

    if source == "daily_prices":
        df = pd.read_parquet(DATA_DIR / "daily_prices/", columns=["ticker"])
    elif source == "fundamentals":
        df = pd.read_parquet(DATA_DIR / "fundamentals.parquet", columns=["ticker"])
    elif source == "both":
        prices = pd.read_parquet(DATA_DIR / "daily_prices/", columns=["ticker"])
        fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet", columns=["ticker"])
        return sorted(set(prices["ticker"]) | set(fund["ticker"]))
    else:
        return []

    return sorted(df["ticker"].unique().tolist())


def run_resumable_job(job_name: str, process_ticker_fn, universe_source: str = "daily_prices",
                      force_reload: bool = False, max_workers: int = 1):
    """
    Run a job with resumability.

    Args:
        job_name: Name of the job (for checkpoint file)
        process_ticker_fn: Function(ticker: str, last_date: date | None) -> date
            Returns the last date processed for that ticker
        universe_source: Data source for universe ("daily_prices", "fundamentals", "both")
        force_reload: If True, ignore checkpoint and reprocess everything
        max_workers: Number of parallel workers (1 = sequential)

    Returns:
        dict with stats: {"processed": N, "skipped": N, "failed": N, "errors": [...]}
    """
    checkpoint = JobCheckpoint(job_name, universe_source)
    tickers = get_universe_tickers(universe_source)

    print(f"[{job_name}] Universe: {len(tickers)} tickers")

    # Check if we can resume
    can_resume = checkpoint.is_valid(tickers, force_reload)
    if can_resume:
        completed = checkpoint.get_completed_tickers()
        pending = [t for t in tickers if t not in completed]
        print(f"[{job_name}] Resuming: {len(completed)} done, {len(pending)} remaining")
    else:
        print(f"[{job_name}] Fresh run (checkpoint invalid or missing)")
        checkpoint.initialize(tickers)
        pending = tickers

    stats = {"processed": 0, "skipped": len(tickers) - len(pending), "failed": 0, "errors": []}

    if max_workers == 1:
        # Sequential
        for ticker in pending:
            try:
                last_date = checkpoint.get_ticker_last_date(ticker)
                checkpoint.mark_ticker_started(ticker)
                result_date = process_ticker_fn(ticker, last_date)
                checkpoint.mark_ticker_complete(ticker, result_date)
                stats["processed"] += 1
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"{ticker}: {e}")
                checkpoint.mark_ticker_failed(ticker, str(e))
    else:
        # Parallel (thread-based for I/O bound tasks)
        import concurrent.futures

        def process_one(ticker):
            try:
                last_date = checkpoint.get_ticker_last_date(ticker)
                checkpoint.mark_ticker_started(ticker)
                result_date = process_ticker_fn(ticker, last_date)
                checkpoint.mark_ticker_complete(ticker, result_date)
                return (ticker, True, None)
            except Exception as e:
                checkpoint.mark_ticker_failed(ticker, str(e))
                return (ticker, False, str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as ex:
            futures = {ex.submit(process_one, t): t for t in pending}
            for fu in concurrent.futures.as_completed(futures):
                ticker, ok, err = fu.result()
                if ok:
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"{ticker}: {err}")

    checkpoint.finalize()
    print(f"[{job_name}] Done: processed={stats['processed']}, skipped={stats['skipped']}, failed={stats['failed']}")
    return stats
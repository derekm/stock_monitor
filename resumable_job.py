#!/usr/bin/env python3
"""
Resumable Job Framework — Checkpoint mechanism for full-universe full-history jobs.

Key features:
- Checkpoint files in backfill_checkpoints/ directory
- Detects new backfill runs that invalidate checkpoints
- Per-ticker progress tracking with last processed date
- Supports forced full-reload via --full-reload flag
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

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class JobCheckpoint:
    """
    Manages checkpoint state for a daily automation job.
    
    Checkpoint file structure:
    {
        "job_name": "rolling_window_analysis",
        "schema_version": 1,
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
        "failed_tickers": []
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
            files_to_hash = ["daily_prices.parquet"]
        elif self.universe_source == "fundamentals":
            files_to_hash = ["fundamentals.parquet"]
        elif self.universe_source == "both":
            files_to_hash = ["daily_prices.parquet", "fundamentals.parquet"]
        
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
            
            # Validate schema
            if state.get("schema_version") != 1:
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
            "schema_version": 1,
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
            with open(self.checkpoint_file, 'w') as f:
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
        df = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["ticker"])
    elif source == "fundamentals":
        df = pd.read_parquet(DATA_DIR / "fundamentals.parquet", columns=["ticker"])
    elif source == "both":
        prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["ticker"])
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
    if checkpoint.is_valid(tickers, force_reload):
        completed = checkpoint.get_completed_tickers()
        print(f"[{job_name}] Resuming: {len(completed)}/{len(tickers)} already complete")
        tickers = [t for t in tickers if t not in completed]
    else:
        print(f"[{job_name}] Starting fresh (force_reload={force_reload})")
        checkpoint.initialize(tickers)
    
    if not tickers:
        print(f"[{job_name}] Nothing to process")
        checkpoint.finalize()
        return {"processed": 0, "skipped": len(get_universe_tickers(universe_source)), "failed": 0, "errors": []}
    
    # Process tickers
    stats = {"processed": 0, "skipped": len(get_universe_tickers(universe_source)) - len(tickers), "failed": 0, "errors": []}
    
    for i, ticker in enumerate(tickers):
        print(f"[{job_name}] [{i+1}/{len(tickers)}] {ticker}...")
        
        checkpoint.mark_ticker_started(ticker)
        
        try:
            last_date = checkpoint.get_ticker_last_date(ticker)
            result_date = process_ticker_fn(ticker, last_date)
            
            checkpoint.mark_ticker_complete(ticker, result_date)
            stats["processed"] += 1
            
        except Exception as e:
            error_msg = f"{ticker}: {str(e)}"
            stats["errors"].append(error_msg)
            stats["failed"] += 1
            checkpoint.mark_ticker_failed(ticker, error_msg)
            print(f"  ERROR: {error_msg}")
    
    checkpoint.finalize()
    print(f"[{job_name}] Complete: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")
    
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: How to use with a daily automation job
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-reload", action="store_true", help="Force full reload")
    parser.add_argument("--job", default="rolling_window_analysis", help="Job name")
    args = parser.parse_args()
    
    def example_process(ticker: str, last_date: date | None) -> date:
        """Example ticker processing function."""
        # This would be replaced with actual job logic
        print(f"  Processing {ticker} from {last_date}")
        return date.today()
    
    stats = run_resumable_job(
        job_name=args.job,
        process_ticker_fn=example_process,
        universe_source="daily_prices",
        force_reload=args.full_reload,
    )
    
    print(f"Stats: {stats}")
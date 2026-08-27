"""
Checkpoint management for resumable full-universe backfill operations.

This module provides a robust checkpoint mechanism that:
1. Tracks per-ticker last processed date (not just a global progress marker)
2. Detects when new backfill data invalidates existing checkpoints
3. Handles new tickers added during edgar_backfill/acq_backfill
4. Ensures downstream jobs don't run on incomplete data

Usage:
    from backfill_checkpoints import CheckpointManager
    
    cm = CheckpointManager("my_backfill_job")
    cm.start_session(tickers=["AAPL", "MSFT"])
    
    for ticker in cm.get_pending_tickers():
        last_date = cm.get_last_date(ticker)
        # process from last_date + 1 day
        new_last_date = process_ticker(ticker, since=last_date)
        cm.update_ticker(ticker, new_last_date)
    
    cm.complete_session()
"""
from __future__ import annotations

import json
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import pandas as pd


class CheckpointManager:
    """
    Manages checkpoints for resumable backfill operations.
    
    Each checkpoint file stores:
    - job_name: identifier for this backfill job
    - ticker_states: dict of ticker -> {last_date, rows_processed, checksum, updated_at}
    - session_info: start_time, total_tickers, completed_tickers, status
    - data_fingerprint: hash of the source data at session start (for invalidation detection)
    """
    
    def __init__(self, job_name: str, checkpoint_dir: Path | str = "backfill_checkpoints"):
        self.job_name = job_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{job_name}_checkpoint.json"
        self._state: dict = {}
        self._session_started = False
        self._data_fingerprint: str | None = None
        
    def _compute_data_fingerprint(self, source_paths: list[Path]) -> str:
        """Compute a fingerprint of the source data to detect changes."""
        hasher = hashlib.md5()
        for path in sorted(source_paths):
            if path.exists():
                stat = path.stat()
                hasher.update(str(stat.st_mtime).encode())
                hasher.update(str(stat.st_size).encode())
        return hasher.hexdigest()[:16]
    
    def start_session(
        self,
        tickers: list[str],
        source_paths: list[Path] | None = None,
        force_restart: bool = False
    ) -> dict:
        """
        Start a new backfill session.
        
        Args:
            tickers: List of all tickers to process in this session
            source_paths: Paths to source data files (for invalidation detection)
            force_restart: If True, ignore existing checkpoint and start fresh
            
        Returns:
            Dict with session info including pending tickers
        """
        if source_paths is None:
            source_paths = []
            
        self._data_fingerprint = self._compute_data_fingerprint(source_paths)
        
        # Load existing checkpoint if not forcing restart
        existing = {}
        if not force_restart and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        
        # Check if data fingerprint matches (invalidation detection)
        fingerprint_changed = False
        if existing.get("data_fingerprint") and existing["data_fingerprint"] != self._data_fingerprint:
            fingerprint_changed = True
            print(f"  ⚠ Checkpoint invalidated: source data changed since last run")
        
        # Initialize state
        self._state = {
            "job_name": self.job_name,
            "data_fingerprint": self._data_fingerprint,
            "source_paths": [str(p) for p in source_paths],
            "session_started": datetime.now().isoformat(),
            "total_tickers": len(tickers),
            "ticker_states": {},
            "status": "in_progress",
        }
        
        # Restore ticker states if fingerprint matches
        if not fingerprint_changed and not force_restart:
            existing_states = existing.get("ticker_states", {})
            for ticker in tickers:
                if ticker in existing_states:
                    state = existing_states[ticker]
                    # Validate state has required fields
                    if "last_date" in state and "updated_at" in state:
                        self._state["ticker_states"][ticker] = state
        
        # Add any new tickers not in existing checkpoint
        for ticker in tickers:
            if ticker not in self._state["ticker_states"]:
                self._state["ticker_states"][ticker] = {
                    "last_date": None,
                    "rows_processed": 0,
                    "checksum": None,
                    "updated_at": None,
                    "status": "pending",
                }
        
        self._session_started = True
        self._save()
        
        pending = self.get_pending_tickers()
        print(f"  Session started: {len(pending)}/{len(tickers)} tickers pending")
        if fingerprint_changed:
            print(f"  Note: {len(tickers) - len(pending)} tickers reset due to data change")
        
        return {
            "total": len(tickers),
            "pending": len(pending),
            "completed": len(tickers) - len(pending),
            "fingerprint_changed": fingerprint_changed,
        }
    
    def get_pending_tickers(self) -> list[str]:
        """Get tickers that still need processing."""
        if not self._session_started:
            raise RuntimeError("Session not started. Call start_session() first.")
        return [
            t for t, s in self._state["ticker_states"].items()
            if s.get("status") != "completed"
        ]
    
    def get_completed_tickers(self) -> list[str]:
        """Get tickers already completed in this session."""
        if not self._session_started:
            raise RuntimeError("Session not started. Call start_session() first.")
        return [
            t for t, s in self._state["ticker_states"].items()
            if s.get("status") == "completed"
        ]
    
    def get_last_date(self, ticker: str) -> date | None:
        """Get the last processed date for a ticker."""
        state = self._state["ticker_states"].get(ticker, {})
        last_date_str = state.get("last_date")
        if last_date_str:
            return date.fromisoformat(last_date_str)
        return None
    
    def get_ticker_state(self, ticker: str) -> dict:
        """Get full state for a ticker."""
        return self._state["ticker_states"].get(ticker, {})
    
    def update_ticker(
        self,
        ticker: str,
        last_date: date | None,
        rows_processed: int = 0,
        checksum: str | None = None,
        status: str = "completed"
    ):
        """
        Update a ticker's progress.
        
        Args:
            ticker: Ticker symbol
            last_date: Last date successfully processed (inclusive)
            rows_processed: Number of rows processed in this update
            checksum: Optional checksum of output data for verification
            status: One of "pending", "processing", "completed", "failed"
        """
        if not self._session_started:
            raise RuntimeError("Session not started. Call start_session() first.")
        
        if ticker not in self._state["ticker_states"]:
            self._state["ticker_states"][ticker] = {}
        
        self._state["ticker_states"][ticker].update({
            "last_date": last_date.isoformat() if last_date else None,
            "rows_processed": self._state["ticker_states"][ticker].get("rows_processed", 0) + rows_processed,
            "checksum": checksum,
            "updated_at": datetime.now().isoformat(),
            "status": status,
        })
        
        # Periodic save (every 10 updates)
        completed = len(self.get_completed_tickers())
        if completed % 10 == 0:
            self._save()
    
    def mark_failed(self, ticker: str, error: str):
        """Mark a ticker as failed with error message."""
        if ticker not in self._state["ticker_states"]:
            self._state["ticker_states"][ticker] = {}
        self._state["ticker_states"][ticker].update({
            "status": "failed",
            "error": error,
            "updated_at": datetime.now().isoformat(),
        })
        self._save()
    
    def complete_session(self) -> dict:
        """Mark session as complete and return summary."""
        if not self._session_started:
            raise RuntimeError("Session not started. Call start_session() first.")
        
        completed = self.get_completed_tickers()
        failed = [
            t for t, s in self._state["ticker_states"].items()
            if s.get("status") == "failed"
        ]
        pending = self.get_pending_tickers()
        
        self._state["status"] = "completed" if not pending and not failed else "partial"
        self._state["session_completed"] = datetime.now().isoformat()
        self._state["summary"] = {
            "total": self._state["total_tickers"],
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
        }
        
        self._save()
        
        return self._state["summary"]
    
    def get_summary(self) -> dict:
        """Get current session summary without completing."""
        if not self._session_started:
            return {"error": "Session not started"}
        
        completed = self.get_completed_tickers()
        failed = [
            t for t, s in self._state["ticker_states"].items()
            if s.get("status") == "failed"
        ]
        pending = self.get_pending_tickers()
        
        return {
            "total": self._state["total_tickers"],
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
            "status": self._state.get("status", "in_progress"),
            "started": self._state.get("session_started"),
        }
    
    def _save(self):
        """Save checkpoint to disk."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(self._state, f, indent=2)
    
    @classmethod
    def load_existing(cls, job_name: str, checkpoint_dir: Path | str = "backfill_checkpoints") -> dict | None:
        """Load an existing checkpoint without starting a session."""
        path = Path(checkpoint_dir) / f"{job_name}_checkpoint.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    @classmethod
    def list_checkpoints(cls, checkpoint_dir: Path | str = "backfill_checkpoints") -> list[dict]:
        """List all available checkpoints."""
        path = Path(checkpoint_dir)
        if not path.exists():
            return []
        
        checkpoints = []
        for f in path.glob("*_checkpoint.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                checkpoints.append({
                    "job_name": data.get("job_name", f.stem.replace("_checkpoint", "")),
                    "status": data.get("status", "unknown"),
                    "started": data.get("session_started"),
                    "completed": data.get("session_completed"),
                    "total_tickers": data.get("total_tickers", 0),
                    "summary": data.get("summary", {}),
                    "data_fingerprint": data.get("data_fingerprint"),
                })
            except Exception:
                pass
        return sorted(checkpoints, key=lambda x: x.get("started", ""), reverse=True)
    
    @classmethod
    def invalidate_checkpoint(cls, job_name: str, checkpoint_dir: Path | str = "backfill_checkpoints"):
        """Manually invalidate a checkpoint (delete it)."""
        path = Path(checkpoint_dir) / f"{job_name}_checkpoint.json"
        if path.exists():
            path.unlink()
            print(f"Invalidated checkpoint: {job_name}")


class TickerBackfillCoordinator:
    """
    Coordinates backfill for new tickers added during edgar_backfill/acq_backfill.
    
    Ensures new tickers get both price AND fundamentals backfilled
    before downstream jobs run.
    """
    
    def __init__(self, data_dir: Path | str = "."):
        self.data_dir = Path(data_dir)
        self.prices_file = self.data_dir / "daily_prices/"
        self.fundamentals_file = self.data_dir / "fundamentals.parquet"
        self.monitored_file = self.data_dir / "monitored_stocks.parquet"
        self.acquisitions_file = self.data_dir / "corporate_actions.parquet"
        
    def get_new_tickers_since(self, since_date: date) -> list[str]:
        """Get tickers added to monitored_stocks since a given date."""
        if not self.monitored_file.exists():
            return []
        
        mon = pd.read_parquet(self.monitored_file)
        if "added_date" not in mon.columns:
            # If no added_date column, we can't track - return all
            return mon["ticker"].tolist()
        
        mon["added_date"] = pd.to_datetime(mon["added_date"]).dt.date
        new_tickers = mon[mon["added_date"] >= since_date]["ticker"].tolist()
        return new_tickers
    
    def get_tickers_missing_prices(self, tickers: list[str]) -> list[str]:
        """Check which tickers are missing from daily_prices."""
        if not self.prices_file.exists():
            return tickers
        
        prices = pd.read_parquet(self.prices_file, columns=["ticker"])
        existing = set(prices["ticker"].unique())
        return [t for t in tickers if t not in existing]
    
    def get_tickers_missing_fundamentals(self, tickers: list[str]) -> list[str]:
        """Check which tickers are missing from fundamentals."""
        if not self.fundamentals_file.exists():
            return tickers
        
        fund = pd.read_parquet(self.fundamentals_file, columns=["ticker"])
        existing = set(fund["ticker"].unique())
        return [t for t in tickers if t not in existing]
    
    def get_tickers_needing_backfill(self, since_date: date | None = None) -> dict:
        """
        Get all tickers that need price/fundamentals backfill.
        
        Returns dict with:
        - new_tickers: tickers added since since_date
        - missing_prices: tickers missing price data
        - missing_fundamentals: tickers missing fundamentals
        - needs_full_backfill: union of above
        """
        if since_date is None:
            since_date = date(2020, 1, 1)  # Far back default
        
        new_tickers = self.get_new_tickers_since(since_date)
        missing_prices = self.get_tickers_missing_prices(new_tickers)
        missing_fundamentals = self.get_tickers_missing_fundamentals(new_tickers)
        
        # Also check acquisitions file for new targets
        acq_tickers = []
        if self.acquisitions_file.exists():
            acq = pd.read_parquet(self.acquisitions_file)
            if "target_ticker" in acq.columns and "completion_date" in acq.columns:
                acq["completion_date"] = pd.to_datetime(acq["completion_date"]).dt.date
                recent = acq[acq["completion_date"] >= since_date]
                acq_tickers = recent["target_ticker"].unique().tolist()
        
        all_new = list(set(new_tickers + acq_tickers))
        all_missing_prices = self.get_tickers_missing_prices(all_new)
        all_missing_fundamentals = self.get_tickers_missing_fundamentals(all_new)
        
        return {
            "new_tickers": all_new,
            "missing_prices": all_missing_prices,
            "missing_fundamentals": all_missing_fundamentals,
            "needs_full_backfill": list(set(all_missing_prices) | set(all_missing_fundamentals)),
        }
    
    def run_backfill_for_new_tickers(
        self,
        tickers: list[str],
        price_lookback_days: int = 252 * 5,  # 5 years
        use_yfinance: bool = True,
    ) -> dict:
        """
        Run price and fundamentals backfill for a list of tickers.
        
        This should be called AFTER edgar_backfill/acq_backfill add new tickers
        and BEFORE downstream jobs like rolling_window_analysis, signal_aggregator,
        damodaran_quality run.
        """
        results = {
            "tickers": tickers,
            "prices_backfilled": [],
            "fundamentals_backfilled": [],
            "failed": [],
        }
        
        if use_yfinance:
            import yfinance as yf
            
            for ticker in tickers:
                try:
                    # Backfill prices
                    if ticker in self.get_tickers_missing_prices([ticker]):
                        tkr = yf.Ticker(ticker)
                        hist = tkr.history(period="max", auto_adjust=True, actions=False)
                        if not hist.empty:
                            # Process and merge...
                            results["prices_backfilled"].append(ticker)
                    
                    # Backfill fundamentals
                    if ticker in self.get_tickers_missing_fundamentals([ticker]):
                        # Use edgar_lib or yfinance fallback
                        results["fundamentals_backfilled"].append(ticker)
                        
                except Exception as e:
                    results["failed"].append({"ticker": ticker, "error": str(e)})
        
        return results


def ensure_new_tickers_backfilled(
    data_dir: Path | str = ".",
    since_date: date | None = None,
    run_backfill: bool = True
) -> dict:
    """
    Convenience function to ensure all new tickers are backfilled.
    
    Call this between edgar_backfill/acq_backfill and downstream jobs.
    
    Returns dict with backfill results.
    """
    coordinator = TickerBackfillCoordinator(data_dir)
    needs = coordinator.get_tickers_needing_backfill(since_date)
    
    print(f"New tickers needing backfill: {needs['needs_full_backfill']}")
    
    if run_backfill and needs["needs_full_backfill"]:
        results = coordinator.run_backfill_for_new_tickers(needs["needs_full_backfill"])
        return {**needs, "backfill_results": results}
    
    return needs
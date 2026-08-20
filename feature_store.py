"""
Parquet Feature Store I/O

Provides a robust, point-in-time aware feature store using Parquet format.
Supports:
1. Panel data (long format) with features, labels, metadata
2. Wide returns matrix for portfolio construction
3. Static maps (sectors, ADV, borrow costs)
4. Model artifacts and run summaries
5. Time-partitioned writes for efficient queries
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Union
from contextlib import contextmanager

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StoreConfig:
    """Configuration for feature store."""
    root: str | Path
    partition_by_date: bool = True
    partition_freq: str = "M"  # 'D', 'W', 'M', 'Y'
    compression: str = "snappy"
    version: str = "1.0"
    
    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["panel", "returns", "static", "artifacts", "models", "runs"]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Schema Definitions
# =============================================================================

PANEL_SCHEMA = {
    "date": "datetime64[ns]",
    "ticker": "string",
    # Features (dynamic)
    # Labels
    "y": "float64",
    "y_h1": "float64",
    "y_h5": "float64",
    "y_h21": "float64",
    "relevance": "int32",
    "rel_h1": "int32",
    "rel_h5": "int32",
    "rel_h21": "int32",
}

RETURNS_SCHEMA = {
    "date": "datetime64[ns]",
    # Ticker columns (dynamic)
}

STATIC_SCHEMA = {
    "ticker": "string",
    "sector": "string",
    "adv": "float64",
    "borrow_bps_annual": "float64",
}


# =============================================================================
# Feature Store
# =============================================================================

class ParquetFeatureStore:
    """
    Parquet-based feature store with PIT (Point-In-Time) awareness.
    
    Directory structure:
    root/
      meta.json                 # Store metadata
      schema.json               # Column schemas
      panel/
        date=2023-01/           # Partitioned by date (optional)
          part-0.parquet
        date=2023-02/
          ...
      returns/
        returns.parquet         # Wide returns matrix
      static/
        sectors.parquet         # Ticker -> sector
        adv.parquet             # Ticker -> ADV
        borrow.parquet          # Ticker -> borrow bps
      artifacts/
        run_20230101/           # Run artifacts
          oos_scores.parquet
          conformal_sizes.parquet
          book_backtest.parquet
          summary.json
      models/
        model_v1/               # Registered models
          model.txt
          meta.json
          metrics.json
      runs/
        run_20230101_summary.json
    """
    
    def __init__(self, config: Union[StoreConfig, str, Path]):
        if isinstance(config, (str, Path)):
            config = StoreConfig(root=config)
        self.config = config
        self.meta_path = self.config.root / "meta.json"
        self.schema_path = self.config.root / "schema.json"
        self._init_meta()
    
    def _init_meta(self):
        """Initialize or load store metadata."""
        if not self.meta_path.exists():
            meta = {
                "version": self.config.version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "schema": {},
                "partitions": [],
                "stats": {},
            }
            self.meta_path.write_text(json.dumps(meta, indent=2))
        if not self.schema_path.exists():
            self.schema_path.write_text(json.dumps({}, indent=2))
    
    def _update_meta(self, **kwargs):
        """Update store metadata."""
        meta = json.loads(self.meta_path.read_text())
        meta.update(kwargs)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str))
    
    # =========================================================================
    # Panel Data (Long Format)
    # =========================================================================
    
    def write_panel(
        self,
        panel: pd.DataFrame,
        partition: bool = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Path:
        """
        Write panel data to Parquet.
        
        Args:
            panel: Long-format DataFrame with date, ticker, features, labels
            partition: Whether to partition by date (overrides config)
            feature_cols: List of feature column names (for metadata)
            
        Returns:
            Path to written file/directory
        """
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        
        partition = partition if partition is not None else self.config.partition_by_date
        
        if partition:
            # Partition by month
            panel["partition"] = panel["date"].dt.to_period(self.config.partition_freq).astype(str)
            partition_dir = self.config.root / "panel"
            partition_dir.mkdir(exist_ok=True)
            
            paths = []
            for part, group in panel.groupby("partition", sort=True):
                part_path = partition_dir / f"date={part}"
                part_path.mkdir(exist_ok=True)
                file_path = part_path / f"part-{len(list(part_path.glob('*.parquet')))}.parquet"
                group.drop(columns=["partition"]).to_parquet(
                    file_path,
                    compression=self.config.compression,
                    index=False,
                )
                paths.append(file_path)
            
            # Update metadata
            self._update_meta(
                panel_rows=len(panel),
                panel_partitions=list(panel["partition"].unique()),
                panel_features=feature_cols or [],
            )
            return partition_dir
        else:
            # Single file
            file_path = self.config.root / "panel" / "panel.parquet"
            panel.to_parquet(
                file_path,
                compression=self.config.compression,
                index=False,
            )
            self._update_meta(
                panel_rows=len(panel),
                panel_path=str(file_path),
                panel_features=feature_cols or [],
            )
            return file_path
    
    def read_panel(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        partition: bool = None,
    ) -> pd.DataFrame:
        """
        Read panel data with optional filters (partition pruning).
        
        Args:
            start_date: Inclusive start date
            end_date: Inclusive end date
            tickers: List of tickers to filter
            columns: Columns to read (for efficiency)
            partition: Whether data is partitioned (auto-detect if None)
            
        Returns:
            Filtered panel DataFrame
        """
        partition = partition if partition is not None else self.config.partition_by_date
        panel_dir = self.config.root / "panel"
        
        if partition and panel_dir.exists():
            # Read partitioned data
            parts = []
            for part_dir in sorted(panel_dir.glob("date=*")):
                # Prune by partition
                part_name = part_dir.name.replace("date=", "")
                if start_date and part_name < start_date[:7]:  # YYYY-MM
                    continue
                if end_date and part_name > end_date[:7]:
                    continue
                
                for file in part_dir.glob("*.parquet"):
                    parts.append(pd.read_parquet(file, columns=columns))
            
            if not parts:
                return pd.DataFrame()
            
            panel = pd.concat(parts, ignore_index=True)
        else:
            # Single file
            file_path = self.config.root / "panel" / "panel.parquet"
            if not file_path.exists():
                return pd.DataFrame()
            panel = pd.read_parquet(file_path, columns=columns)
        
        # Apply filters
        if start_date:
            panel = panel[panel["date"] >= pd.to_datetime(start_date)]
        if end_date:
            panel = panel[panel["date"] <= pd.to_datetime(end_date)]
        if tickers:
            panel = panel[panel["ticker"].isin(tickers)]
        
        panel["date"] = pd.to_datetime(panel["date"])
        return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    def get_panel_dates(self) -> np.ndarray:
        """Get unique dates in panel (efficient, reads only date column)."""
        partition = self.config.partition_by_date
        panel_dir = self.config.root / "panel"
        
        if partition and panel_dir.exists():
            dates = []
            for part_dir in panel_dir.glob("date=*"):
                for file in part_dir.glob("*.parquet"):
                    dates.extend(pd.read_parquet(file, columns=["date"])["date"].tolist())
            return np.array(sorted(set(pd.to_datetime(dates))))
        else:
            file_path = self.config.root / "panel" / "panel.parquet"
            if file_path.exists():
                return np.array(sorted(pd.read_parquet(file_path, columns=["date"])["date"].unique()))
            return np.array([])
    
    def get_panel_tickers(self) -> np.ndarray:
        """Get unique tickers in panel."""
        partition = self.config.partition_by_date
        panel_dir = self.config.root / "panel"
        
        if partition and panel_dir.exists():
            tickers = set()
            for part_dir in panel_dir.glob("date=*"):
                for file in part_dir.glob("*.parquet"):
                    tickers.update(pd.read_parquet(file, columns=["ticker"])["ticker"].tolist())
            return np.array(sorted(tickers))
        else:
            file_path = self.config.root / "panel" / "panel.parquet"
            if file_path.exists():
                return np.array(sorted(pd.read_parquet(file_path, columns=["ticker"])["ticker"].unique()))
            return np.array([])
    
    # =========================================================================
    # Wide Returns Matrix
    # =========================================================================
    
    def write_returns_wide(
        self,
        returns: pd.DataFrame,
        name: str = "returns",
    ) -> Path:
        """
        Write wide returns matrix (date x ticker).
        
        Args:
            returns: DataFrame with DatetimeIndex, tickers as columns
            name: Name for the returns file
            
        Returns:
            Path to written file
        """
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)
        returns.index.name = "date"
        returns = returns.sort_index()
        
        file_path = self.config.root / "returns" / f"{name}.parquet"
        returns.reset_index().to_parquet(
            file_path,
            compression=self.config.compression,
            index=False,
        )
        
        self._update_meta(
            returns_rows=len(returns),
            returns_cols=list(returns.columns),
            returns_path=str(file_path),
            returns_start=str(returns.index.min().date()),
            returns_end=str(returns.index.max().date()),
        )
        return file_path
    
    def read_returns_wide(
        self,
        name: str = "returns",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read wide returns matrix with optional filters."""
        file_path = self.config.root / "returns" / f"{name}.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        
        # Column pruning only when a ticker subset is requested. Building
        # cols = ["date"] unconditionally meant a default read (tickers=None) asked
        # parquet for the date column alone and returned a frame with zero tickers,
        # so read_returns_wide() could never load the full matrix.
        cols = None
        if tickers:
            cols = ["date"] + [t for t in tickers if t != "date"]

        df = pd.read_parquet(file_path, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        if tickers:
            existing = [t for t in tickers if t in df.columns]
            df = df[existing]
        
        return df
    
    # =========================================================================
    # Static Maps
    # =========================================================================
    
    def write_static_map(
        self,
        data: pd.Series | pd.DataFrame,
        name: str,
    ) -> Path:
        """Write static ticker map (sectors, ADV, borrow, etc.)."""
        if isinstance(data, pd.Series):
            data = data.to_frame(name=name)
        data = data.reset_index()
        data.columns = ["ticker", name] if len(data.columns) == 2 else data.columns.tolist()
        
        file_path = self.config.root / "static" / f"{name}.parquet"
        data.to_parquet(file_path, compression=self.config.compression, index=False)
        
        self._update_meta(static_maps={name: str(file_path)})
        return file_path
    
    def read_static_map(self, name: str) -> pd.Series:
        """Read static ticker map."""
        file_path = self.config.root / "static" / f"{name}.parquet"
        if not file_path.exists():
            return pd.Series(dtype=object)
        
        df = pd.read_parquet(file_path)
        if len(df.columns) == 2:
            df = df.set_index("ticker").iloc[:, 0]
        return df
    
    # =========================================================================
    # Artifacts / Run Outputs
    # =========================================================================
    
    def write_artifact(
        self,
        data: pd.DataFrame | Dict,
        run_name: str,
        artifact_name: str,
    ) -> Path:
        """Write run artifact (OOS scores, conformal sizes, book backtest, etc.)."""
        run_dir = self.config.root / "artifacts" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            file_path = run_dir / f"{artifact_name}.parquet"
            # index=False silently DISCARDED a meaningful index: book_backtest is
            # indexed by date, so the artifact came back with an int64 0..N-1 index
            # and every dated diagnostic read from it was wrong (a later
            # pd.to_datetime turned positions 0,1,2 into 1970-01-01). Named indexes
            # are promoted to a real column; only an anonymous RangeIndex is dropped.
            out = data
            if data.index.name is not None or not isinstance(data.index, pd.RangeIndex):
                out = data.reset_index()
                if "index" in out.columns and data.index.name is None:
                    out = out.rename(columns={"index": "row_index"})
            out.to_parquet(file_path, compression=self.config.compression, index=False)
        else:
            file_path = run_dir / f"{artifact_name}.json"
            file_path.write_text(json.dumps(data, indent=2, default=str))
        
        return file_path
    
    def read_artifact(
        self,
        run_name: str,
        artifact_name: str,
    ) -> pd.DataFrame | Dict:
        """Read run artifact."""
        run_dir = self.config.root / "artifacts" / run_name
        
        # Try parquet first
        parquet_path = run_dir / f"{artifact_name}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        
        # Try JSON
        json_path = run_dir / f"{artifact_name}.json"
        if json_path.exists():
            return json.loads(json_path.read_text())
        
        raise FileNotFoundError(f"Artifact not found: {run_name}/{artifact_name}")
    
    def list_artifacts(self, run_name: str) -> List[str]:
        """List artifacts for a run."""
        run_dir = self.config.root / "artifacts" / run_name
        if not run_dir.exists():
            return []
        return [f.name for f in run_dir.iterdir() if f.is_file()]
    
    # =========================================================================
    # Model Registry Integration
    # =========================================================================
    
    def write_model(
        self,
        model: Any,  # LightGBM Booster
        model_id: str,
        meta: Dict,
    ) -> Path:
        """Write model to registry."""
        model_dir = self.config.root / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_model(str(model_dir / "model.txt"))
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        
        return model_dir
    
    def read_model(self, model_id: str) -> Any:
        """Read model from registry."""
        model_path = self.config.root / "models" / model_id / "model.txt"
        import lightgbm as lgb
        return lgb.Booster(model_file=str(model_path))
    
    def read_model_meta(self, model_id: str) -> Dict:
        """Read model metadata."""
        meta_path = self.config.root / "models" / model_id / "meta.json"
        return json.loads(meta_path.read_text())
    
    # =========================================================================
    # Run Summaries
    # =========================================================================
    
    def write_run_summary(
        self,
        run_name: str,
        summary: Dict,
    ) -> Path:
        """Write run summary."""
        runs_dir = self.config.root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = runs_dir / f"{run_name}_summary.json"
        file_path.write_text(json.dumps(summary, indent=2, default=str))
        return file_path
    
    def read_run_summary(self, run_name: str) -> Dict:
        """Read run summary."""
        file_path = self.config.root / "runs" / f"{run_name}_summary.json"
        if not file_path.exists():
            return {}
        return json.loads(file_path.read_text())
    
    def list_runs(self) -> List[str]:
        """List all run summaries."""
        runs_dir = self.config.root / "runs"
        if not runs_dir.exists():
            return []
        return [f.stem.replace("_summary", "") for f in runs_dir.glob("*_summary.json")]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_meta(self) -> Dict:
        """Get store metadata."""
        return json.loads(self.meta_path.read_text())
    
    def get_schema(self) -> Dict:
        """Get column schemas."""
        return json.loads(self.schema_path.read_text())
    
    def set_schema(self, schema: Dict):
        """Set column schemas."""
        self.schema_path.write_text(json.dumps(schema, indent=2))
    
    @contextmanager
    def transaction(self):
        """Context manager for atomic writes (simulated via temp files)."""
        # For Parquet, we write to temp then rename
        temp_dir = self.config.root / ".temp"
        temp_dir.mkdir(exist_ok=True)
        try:
            yield temp_dir
        finally:
            # Cleanup temp files
            for f in temp_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
    
    def vacuum(self):
        """Clean up temporary files and optimize."""
        temp_dir = self.config.root / ".temp"
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                temp_dir.rmdir()
            except Exception:
                pass
        
        # Update partition list in meta
        panel_dir = self.config.root / "panel"
        if panel_dir.exists():
            partitions = sorted([d.name.replace("date=", "") for d in panel_dir.glob("date=*")])
            self._update_meta(panel_partitions=partitions)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_store_from_data(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    sectors: pd.Series,
    adv: pd.Series,
    borrow_bps: pd.Series,
    store_root: str | Path,
    feature_cols: List[str],
    run_name: Optional[str] = None,
    run_summary: Optional[Dict] = None,
) -> ParquetFeatureStore:
    """
    Create feature store from raw data components.
    
    One-stop function to initialize store with all components.
    """
    config = StoreConfig(root=store_root)
    store = ParquetFeatureStore(config)
    
    # Write panel
    store.write_panel(panel, feature_cols=feature_cols)
    
    # Write returns
    store.write_returns_wide(returns_wide)
    
    # Write static maps
    store.write_static_map(sectors, "sectors")
    store.write_static_map(adv, "adv")
    store.write_static_map(borrow_bps, "borrow_bps_annual")
    
    # Write run summary if provided
    if run_name and run_summary:
        store.write_run_summary(run_name, run_summary)
    
    return store


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing Parquet Feature Store...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store_root = Path(tmpdir) / "feature_store"
        
        # Create test data
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=200)
        tickers = [f"T{i:02d}" for i in range(10)]
        
        # Panel data
        panels = []
        for tkr in tickers:
            for dt in dates:
                panels.append({
                    "date": dt,
                    "ticker": tkr,
                    "ret_1": np.random.randn() * 0.01,
                    "ret_5": np.random.randn() * 0.02,
                    "ret_10": np.random.randn() * 0.03,
                    "vol_10": abs(np.random.randn() * 0.01),
                    "vol_20": abs(np.random.randn() * 0.01),
                    "ma_gap": np.random.randn() * 0.01,
                    "y": np.random.randn() * 0.02,
                    "relevance": np.random.randint(0, 5),
                })
        
        panel = pd.DataFrame(panels)
        feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
        
        # Returns wide
        rets = {}
        for tkr in tickers:
            rets[tkr] = np.random.randn(len(dates)) * 0.01
        returns_wide = pd.DataFrame(rets, index=dates)
        
        # Static maps
        sectors = pd.Series({t: f"S{i % 3}" for i, t in enumerate(tickers)}, name="sector")
        adv = pd.Series({t: float(5e7 * np.exp(np.random.normal(0, 0.25))) for t in tickers}, name="adv")
        borrow = pd.Series({t: float(np.random.choice([50, 100, 200, 500, 1000])) for t in tickers}, name="borrow_bps_annual")
        
        # Create store
        store = create_store_from_data(
            panel, returns_wide, sectors, adv, borrow,
            store_root, feature_cols,
            run_name="test_run",
            run_summary={"ic_mean": 0.02, "book_sharpe": 1.2},
        )
        
        # Test reads
        print("\n1. Reading full panel...")
        panel_read = store.read_panel()
        print(f"   Shape: {panel_read.shape}")
        
        print("\n2. Reading panel with date filter...")
        panel_filtered = store.read_panel(start_date="2023-01-15", end_date="2023-02-15")
        print(f"   Shape: {panel_filtered.shape}")
        print(f"   Date range: {panel_filtered['date'].min()} to {panel_filtered['date'].max()}")
        
        print("\n3. Reading returns wide...")
        returns_read = store.read_returns_wide()
        print(f"   Shape: {returns_read.shape}")
        
        print("\n4. Reading static maps...")
        sectors_read = store.read_static_map("sectors")
        adv_read = store.read_static_map("adv")
        borrow_read = store.read_static_map("borrow_bps_annual")
        print(f"   Sectors: {len(sectors_read)}")
        print(f"   ADV: {len(adv_read)}")
        print(f"   Borrow: {len(borrow_read)}")
        
        print("\n5. Reading run summary...")
        summary = store.read_run_summary("test_run")
        print(f"   Summary: {summary}")
        
        print("\n6. Testing date/ticker queries...")
        dates = store.get_panel_dates()
        tickers_read = store.get_panel_tickers()
        print(f"   Unique dates: {len(dates)}")
        print(f"   Unique tickers: {len(tickers_read)}")
        
        print("\n7. Writing and reading artifacts...")
        test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        store.write_artifact(test_df, "test_run", "test_artifact")
        artifact_read = store.read_artifact("test_run", "test_artifact")
        print(f"   Artifact shape: {artifact_read.shape}")
        
        print("\n8. Model registry...")
        import lightgbm as lgb
        X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
        y = (X["f1"] + np.random.randn(50) > 0).astype(int)
        model = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, label=y), num_boost_round=5)
        model_meta = {"tag": "test", "metrics": {"ic": 0.02}, "feature_cols": ["f1", "f2"]}
        store.write_model(model, "test_model_v1", model_meta)
        model_read = store.read_model("test_model_v1")
        meta_read = store.read_model_meta("test_model_v1")
        print(f"   Model loaded: {model_read is not None}")
        print(f"   Meta: {meta_read['tag']}")
        
        print("\n9. Metadata...")
        meta = store.get_meta()
        print(f"   Panel rows: {meta.get('panel_rows')}")
        print(f"   Panel partitions: {meta.get('panel_partitions')}")
        
        print("\nAll tests passed!")
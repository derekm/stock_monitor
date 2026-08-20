"""
Champion-Challenger Registry with Shadow Gate

Implements:
1. Model registry with challenger/champion states
2. Shadow book evaluation (paper trading) before promotion
3. Promotion/demotion gates with metrics thresholds
4. Audit trail and history logging
5. Kill-switch integration
"""

from __future__ import annotations

import json
import hashlib
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelMetrics:
    """Standardized metrics for model comparison."""
    ic_mean: float = np.nan
    ic_ir: float = np.nan
    book_sharpe: float = np.nan
    gross_sharpe: float = np.nan
    avg_cost: float = np.nan
    avg_borrow: float = np.nan
    avg_turnover: float = np.nan
    max_drawdown: float = np.nan
    avg_book_beta: float = np.nan
    cum_net: float = np.nan
    n_oos_rows: int = 0
    n_test_days: int = 0
    
    def to_dict(self) -> dict:
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in asdict(self).items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelMetrics':
        return cls(**{k: float(v) if isinstance(v, (int, float)) and k != 'n_oos_rows' and k != 'n_test_days' else v 
                      for k, v in d.items()})


@dataclass
class ModelMetadata:
    """Model metadata stored in registry."""
    model_id: str
    tag: str
    status: str  # challenger, champion, retired, killed, rejected
    metrics: ModelMetrics
    params: dict
    feature_cols: list[str]
    extra: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: Optional[str] = None
    demoted_at: Optional[str] = None
    demote_reason: Optional[str] = None


@dataclass
class PromotionGate:
    """Gate requirements for promotion."""
    min_ic_mean: float = -np.inf
    min_ic_ir: float = -np.inf
    min_book_sharpe: float = -np.inf
    max_drawdown: float = np.inf
    max_avg_cost: float = np.inf
    max_avg_turnover: float = np.inf
    require_shadow_clean_days: int = 0
    
    def check(self, challenger: ModelMetrics, champion: Optional[ModelMetrics] = None) -> tuple[bool, str]:
        """Check if challenger passes gates."""
        if challenger.ic_mean < self.min_ic_mean:
            return False, f"IC mean {challenger.ic_mean:.4f} < {self.min_ic_mean:.4f}"
        if challenger.ic_ir < self.min_ic_ir:
            return False, f"IC IR {challenger.ic_ir:.4f} < {self.min_ic_ir:.4f}"
        if challenger.book_sharpe < self.min_book_sharpe:
            return False, f"Book Sharpe {challenger.book_sharpe:.2f} < {self.min_book_sharpe:.2f}"
        if challenger.max_drawdown > self.max_drawdown:
            return False, f"Max DD {challenger.max_drawdown:.2%} > {self.max_drawdown:.2%}"
        if challenger.avg_cost > self.max_avg_cost:
            return False, f"Avg cost {challenger.avg_cost:.5f} > {self.max_avg_cost:.5f}"
        if challenger.avg_turnover > self.max_avg_turnover:
            return False, f"Avg turnover {challenger.avg_turnover:.3f} > {self.max_avg_turnover:.3f}"
        
        # Improvement over champion (if exists)
        if champion is not None:
            if challenger.book_sharpe <= champion.book_sharpe:
                return False, f"Book Sharpe not improved over champion ({challenger.book_sharpe:.2f} <= {champion.book_sharpe:.2f})"
            if challenger.max_drawdown >= champion.max_drawdown:
                return False, f"Max DD not improved over champion ({challenger.max_drawdown:.2%} >= {champion.max_drawdown:.2%})"
        
        return True, "All gates passed"


@dataclass
class ShadowState:
    """State of shadow evaluation."""
    model_id: str
    status: str  # running, clean, failed
    clean_days: int = 0
    breach_days: int = 0
    target_clean_days: int = 20
    history: list = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# Registry
# =============================================================================

class ModelRegistry:
    """
    Champion-challenger model registry with shadow evaluation.
    
    File structure:
    registry/
      index.json          # All models metadata
      champion.json       # Current champion pointer
      history.jsonl       # Audit log
      challengers/
        <model_id>/
          model.txt       # LightGBM model file
          meta.json       # ModelMetadata
          metrics.json    # ModelMetrics
      shadow/
        index.json        # Shadow evaluation states
    """
    
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.registry_dir = self.root / "registry"
        self.challengers_dir = self.registry_dir / "challengers"
        self.shadow_dir = self.root / "shadow"
        
        for d in [self.registry_dir, self.challengers_dir, self.shadow_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.registry_dir / "index.json"
        self.champion_path = self.registry_dir / "champion.json"
        self.history_path = self.registry_dir / "history.jsonl"
        self.shadow_index_path = self.shadow_dir / "index.json"
        
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"models": {}}, indent=2))
        if not self.shadow_index_path.exists():
            self.shadow_index_path.write_text(json.dumps({"shadows": {}}, indent=2))
    
    # ----- Internal helpers -----
    
    def _read_index(self) -> dict:
        return json.loads(self.index_path.read_text())
    
    def _write_index(self, idx: dict):
        self.index_path.write_text(json.dumps(idx, indent=2, default=str))
    
    def _read_shadow_index(self) -> dict:
        return json.loads(self.shadow_index_path.read_text())
    
    def _write_shadow_index(self, idx: dict):
        self.shadow_index_path.write_text(json.dumps(idx, indent=2, default=str))
    
    def _append_history(self, event: dict):
        event = {**event, "ts": datetime.now(timezone.utc).isoformat()}
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    
    @staticmethod
    def _make_model_id(tag: str, metrics: ModelMetrics) -> str:
        h = hashlib.sha1(
            json.dumps({"tag": tag, "metrics": metrics.to_dict()}, sort_keys=True, default=str).encode()
        ).hexdigest()[:10]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{tag}_{stamp}_{h}"
    
    # ----- Challenger registration -----
    
    def register_challenger(
        self,
        model: lgb.Booster,
        tag: str,
        metrics: ModelMetrics,
        params: Optional[dict] = None,
        feature_cols: Optional[list[str]] = None,
        extra: Optional[dict] = None,
    ) -> str:
        """Register a new challenger model."""
        model_id = self._make_model_id(tag, metrics)
        cdir = self.challengers_dir / model_id
        cdir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_model(str(cdir / "model.txt"))
        
        # Save metadata
        meta = ModelMetadata(
            model_id=model_id,
            tag=tag,
            status="challenger",
            metrics=metrics,
            params=params or {},
            feature_cols=feature_cols or [],
            extra=extra or {},
        )
        (cdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2, default=str))
        (cdir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2, default=str))
        
        # Update index
        idx = self._read_index()
        idx["models"][model_id] = {
            "tag": tag,
            "status": "challenger",
            "path": str(cdir),
            "metrics": metrics.to_dict(),
        }
        self._write_index(idx)
        
        self._append_history({
            "event": "register_challenger",
            "model_id": model_id,
            "tag": tag,
            "metrics": metrics.to_dict(),
        })
        
        return model_id
    
    def load_booster(self, model_id: str) -> lgb.Booster:
        """Load a LightGBM booster from registry."""
        path = self.challengers_dir / model_id / "model.txt"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        return lgb.Booster(model_file=str(path))
    
    def load_metadata(self, model_id: str) -> ModelMetadata:
        """Load model metadata."""
        path = self.challengers_dir / model_id / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {model_id}")
        data = json.loads(path.read_text())
        data["metrics"] = ModelMetrics.from_dict(data["metrics"])
        return ModelMetadata(**data)
    
    # ----- Champion management -----
    
    def get_champion(self) -> Optional[ModelMetadata]:
        """Get current champion metadata."""
        if not self.champion_path.exists():
            return None
        data = json.loads(self.champion_path.read_text())
        model_id = data["model_id"]
        # Load full metadata from challenger directory
        return self.load_metadata(model_id)
    
    def get_champion_metrics(self) -> Optional[ModelMetrics]:
        """Get current champion metrics."""
        champ = self.get_champion()
        return champ.metrics if champ else None
    
    def promote(
        self,
        model_id: str,
        reason: str,
        gates: Optional[PromotionGate] = None,
    ) -> dict:
        """
        Promote challenger to champion.
        
        Args:
            model_id: Challenger to promote
            reason: Promotion reason
            gates: Optional PromotionGate to enforce
            
        Returns:
            Dict with promotion result
        """
        idx = self._read_index()
        
        if model_id not in idx["models"]:
            return {"promoted": False, "reason": f"Unknown model_id: {model_id}"}
        
        chal_data = idx["models"][model_id]
        if chal_data["status"] != "challenger":
            return {"promoted": False, "reason": f"Model not a challenger: {chal_data['status']}"}
        
        chal_metrics = ModelMetrics.from_dict(chal_data["metrics"])
        champ = self.get_champion()
        champ_metrics = champ.metrics if champ else None
        
        # Check gates
        if gates:
            passed, msg = gates.check(chal_metrics, champ_metrics)
            if not passed:
                self._append_history({
                    "event": "reject_promotion",
                    "model_id": model_id,
                    "reason": msg,
                    "challenger_metrics": chal_metrics.to_dict(),
                    "champion_metrics": champ_metrics.to_dict() if champ_metrics else None,
                })
                return {"promoted": False, "reason": msg, "gate_check": msg}
        
        # Demote old champion
        if champ is not None:
            old_id = champ.model_id
            if old_id in idx["models"]:
                idx["models"][old_id]["status"] = "retired"
                # Update old meta
                old_meta_path = self.challengers_dir / old_id / "meta.json"
                if old_meta_path.exists():
                    old_meta = json.loads(old_meta_path.read_text())
                    old_meta["status"] = "retired"
                    old_meta_path.write_text(json.dumps(old_meta, indent=2, default=str))
        
        # Promote new champion
        idx["models"][model_id]["status"] = "champion"
        self._write_index(idx)
        
        # Update meta
        meta_path = self.challengers_dir / model_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "champion"
        meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        
        # Write champion pointer
        payload = {
            "model_id": model_id,
            "tag": chal_data["tag"],
            "metrics": chal_metrics.to_dict(),
            "path": chal_data["path"],
            "reason": reason,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self.champion_path.write_text(json.dumps(payload, indent=2, default=str))
        
        self._append_history({
            "event": "promote",
            "model_id": model_id,
            "reason": reason,
            "metrics": chal_metrics.to_dict(),
            "previous_champion": champ.model_id if champ else None,
        })
        
        return {"promoted": True, "champion": payload}
    
    def demote_champion(self, reason: str, to_status: str = "killed") -> dict:
        """Demote current champion."""
        champ = self.get_champion()
        if champ is None:
            return {"demoted": False, "reason": "No champion to demote"}
        
        model_id = champ.model_id
        idx = self._read_index()
        
        if model_id in idx["models"]:
            idx["models"][model_id]["status"] = to_status
            self._write_index(idx)
        
        # Update meta
        meta_path = self.challengers_dir / model_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = to_status
            meta["demoted_at"] = datetime.now(timezone.utc).isoformat()
            meta["demote_reason"] = reason
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        
        # Archive and remove champion pointer
        if self.champion_path.exists():
            killed = {**asdict(champ), "status": to_status, "demote_reason": reason,
                      "demoted_at": datetime.now(timezone.utc).isoformat()}
            (self.registry_dir / "last_killed_champion.json").write_text(
                json.dumps(killed, indent=2, default=str)
            )
            self.champion_path.unlink()
        
        self._append_history({
            "event": "demote_champion",
            "model_id": model_id,
            "reason": reason,
            "to_status": to_status,
        })
        
        return {"demoted": True, "model_id": model_id, "reason": reason}
    
    def reject_challenger(self, model_id: str, reason: str):
        """Reject a challenger."""
        idx = self._read_index()
        if model_id in idx["models"]:
            idx["models"][model_id]["status"] = "rejected"
            self._write_index(idx)
        
        meta_path = self.challengers_dir / model_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "rejected"
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        
        self._append_history({
            "event": "reject_challenger",
            "model_id": model_id,
            "reason": reason,
        })
    
    # ----- Shadow evaluation -----
    
    def start_shadow(
        self,
        model_id: str,
        target_clean_days: int = 20,
    ) -> ShadowState:
        """Start shadow evaluation for a challenger."""
        if model_id not in self._read_index()["models"]:
            raise KeyError(f"Unknown model_id: {model_id}")
        
        idx = self._read_shadow_index()
        state = ShadowState(
            model_id=model_id,
            status="running",
            target_clean_days=target_clean_days,
        )
        idx["shadows"][model_id] = asdict(state)
        self._write_shadow_index(idx)
        
        self._append_history({
            "event": "shadow_start",
            "model_id": model_id,
            "target_clean_days": target_clean_days,
        })
        
        return state
    
    def get_shadow_state(self, model_id: str) -> Optional[ShadowState]:
        """Get shadow evaluation state."""
        idx = self._read_shadow_index()
        if model_id not in idx.get("shadows", {}):
            return None
        return ShadowState(**idx["shadows"][model_id])
    
    def update_shadow_day(
        self,
        model_id: str,
        date: Any,
        rolling_ic: float,
        rolling_sharpe: float,
        drawdown: float,
        turnover: float,
        kill_config: "KillSwitchConfig",
    ) -> ShadowState:
        """Update shadow state with daily monitoring metrics."""
        idx = self._read_shadow_index()
        
        if model_id not in idx.get("shadows", {}):
            # Auto-start if not exists
            self.start_shadow(model_id, target_clean_days=kill_config.shadow_clean_days)
            idx = self._read_shadow_index()
        
        state_data = idx["shadows"][model_id]
        state = ShadowState(**state_data)
        
        # Check for breaches using kill-switch thresholds
        breach = (
            (not np.isnan(rolling_ic) and rolling_ic < kill_config.min_rolling_ic) or
            (not np.isnan(rolling_sharpe) and rolling_sharpe < kill_config.min_rolling_sharpe) or
            (drawdown > kill_config.max_drawdown) or
            (not np.isnan(turnover) and turnover > kill_config.max_avg_turnover)
        )
        
        day_record = {
            "date": str(pd.to_datetime(date).date()),
            "breach": breach,
            "rolling_ic": rolling_ic,
            "rolling_sharpe": rolling_sharpe,
            "drawdown": drawdown,
            "turnover": turnover,
        }
        
        state.history = (state.history or [])[-200:] + [day_record]
        
        if breach:
            state.clean_days = 0
            state.breach_days += 1
            if state.breach_days >= kill_config.strikes_to_kill:
                state.status = "failed"
        else:
            state.clean_days += 1
            if state.clean_days >= state.target_clean_days and state.status != "failed":
                state.status = "clean"
            elif state.status not in ("failed", "clean"):
                state.status = "running"
        
        idx["shadows"][model_id] = asdict(state)
        self._write_shadow_index(idx)
        
        self._append_history({
            "event": "shadow_day",
            "model_id": model_id,
            "day": day_record,
            "status": state.status,
            "clean_days": state.clean_days,
        })
        
        return state
    
    def is_shadow_clean(self, model_id: str, min_clean_days: Optional[int] = None) -> tuple[bool, ShadowState]:
        """Check if shadow evaluation is clean."""
        state = self.get_shadow_state(model_id)
        if state is None:
            return False, ShadowState(model_id=model_id, status="none", clean_days=0)
        
        required = min_clean_days or state.target_clean_days
        is_clean = state.status == "clean" and state.clean_days >= required
        return is_clean, state
    
    # ----- Comparison -----
    
    def compare_to_champion(self, metrics: ModelMetrics, keys: Optional[list[str]] = None) -> dict:
        """Compare metrics to current champion."""
        champ = self.get_champion()
        if champ is None:
            return {"has_champion": False, "decision": "promote_first"}
        
        keys = keys or ["ic_mean", "ic_ir", "book_sharpe", "max_drawdown", "avg_cost", "avg_turnover"]
        champ_metrics = champ.metrics
        
        diff = {}
        for k in keys:
            c_val = getattr(metrics, k, np.nan)
            ch_val = getattr(champ_metrics, k, np.nan)
            diff[k] = float(c_val) - float(ch_val) if not (np.isnan(c_val) or np.isnan(ch_val)) else np.nan
        
        return {
            "has_champion": True,
            "champion_id": champ.model_id,
            "champion_tag": champ.tag,
            "diff": diff,
        }
    
    # ----- Listing -----
    
    def list_models(self, status: Optional[str] = None) -> list[dict]:
        """List all models, optionally filtered by status."""
        idx = self._read_index()
        result = []
        for mid, info in idx["models"].items():
            if status is None or info["status"] == status:
                result.append({"model_id": mid, **info})
        return sorted(result, key=lambda x: x.get("metrics", {}).get("book_sharpe", -np.inf), reverse=True)
    
    def get_history(self, limit: int = 100) -> list[dict]:
        """Get recent history events."""
        if not self.history_path.exists():
            return []
        events = []
        with self.history_path.open("r") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events[-limit:]


# =============================================================================
# Kill Switch Config (for shadow monitoring)
# =============================================================================

@dataclass
class KillSwitchConfig:
    """Kill-switch thresholds (also used for shadow monitoring)."""
    ic_window: int = 42
    sharpe_window: int = 42
    drawdown_window: int = 84
    min_rolling_ic: float = -0.03
    min_rolling_sharpe: float = -0.75
    max_drawdown: float = 0.12
    max_avg_turnover: float = 1.5
    strikes_to_kill: int = 3
    shadow_clean_days: int = 15
    auto_demote: bool = True


# =============================================================================
# Kill Switch Monitor (for live/backtest)
# =============================================================================

class KillSwitchMonitor:
    """Live kill-switch monitor with strike accumulation."""
    
    def __init__(self, registry: ModelRegistry, cfg: Optional[KillSwitchConfig] = None):
        self.registry = registry
        self.cfg = cfg or KillSwitchConfig()
        self.ic_series: list[tuple] = []
        self.ret_series: list[tuple] = []
        self.turn_series: list[tuple] = []
        self.strikes = 0
        self.ok_streak = 0
    
    def _rolling_ic(self, w: int) -> float:
        if len(self.ic_series) < max(8, w // 4):
            return np.nan
        vals = np.array([v for _, v in self.ic_series[-w:]], float)
        return float(np.nanmean(vals))
    
    def _rolling_sharpe(self, w: int) -> float:
        if len(self.ret_series) < max(8, w // 4):
            return np.nan
        r = np.array([v for _, v in self.ret_series[-w:]], float)
        return float(np.nanmean(r) / (np.nanstd(r) + 1e-12) * np.sqrt(252))
    
    def _drawdown(self, w: int) -> float:
        if len(self.ret_series) < 5:
            return 0.0
        r = pd.Series([v for _, v in self.ret_series[-w:]])
        eq = (1.0 + r).cumprod()
        return float((1.0 - eq / eq.cummax()).max())
    
    def _avg_turnover(self, w: int) -> float:
        if not self.turn_series:
            return np.nan
        vals = np.array([v for _, v in self.turn_series[-w:]], float)
        return float(np.nanmean(vals))
    
    def update(
        self,
        date: Any,
        daily_ic: float,
        net_ret: float,
        turnover: float,
        also_update_shadow: bool = True,
        shadow_model_id: Optional[str] = None,
    ) -> dict:
        """Update live monitor and optionally shadow state."""
        date = pd.to_datetime(date)
        self.ic_series.append((date, float(daily_ic)))
        self.ret_series.append((date, float(net_ret)))
        self.turn_series.append((date, float(turnover)))
        
        ric = self._rolling_ic(self.cfg.ic_window)
        rsh = self._rolling_sharpe(self.cfg.sharpe_window)
        dd = self._drawdown(self.cfg.drawdown_window)
        at = self._avg_turnover(self.cfg.sharpe_window)
        
        breach = any([
            (not np.isnan(ric) and ric < self.cfg.min_rolling_ic),
            (not np.isnan(rsh) and rsh < self.cfg.min_rolling_sharpe),
            dd > self.cfg.max_drawdown,
            (not np.isnan(at) and at > self.cfg.max_avg_turnover),
        ])
        
        if breach:
            self.strikes += 1
            self.ok_streak = 0
        else:
            self.ok_streak += 1
            if self.ok_streak >= 10 and self.strikes > 0:
                self.strikes -= 1
                self.ok_streak = 0
        
        out = {
            "date": str(date.date()),
            "rolling_ic": ric,
            "rolling_sharpe": rsh,
            "drawdown": dd,
            "avg_turnover": at,
            "strikes": self.strikes,
            "breach": breach,
            "breach_ic": (not np.isnan(ric) and ric < self.cfg.min_rolling_ic),
            "breach_sharpe": (not np.isnan(rsh) and rsh < self.cfg.min_rolling_sharpe),
            "breach_dd": dd > self.cfg.max_drawdown,
            "breach_turn": (not np.isnan(at) and at > self.cfg.max_avg_turnover),
        }
        
        # Demote champion if strikes exceeded
        demoted = False
        if self.strikes >= self.cfg.strikes_to_kill and self.cfg.auto_demote:
            reasons = []
            if out["breach_ic"]: reasons.append(f"IC={ric:.4f}")
            if out["breach_sharpe"]: reasons.append(f"Sharpe={rsh:.2f}")
            if out["breach_dd"]: reasons.append(f"DD={dd:.2%}")
            if out["breach_turn"]: reasons.append(f"Turn={at:.2f}")
            reason = "kill_switch: " + ", ".join(reasons)
            demote_result = self.registry.demote_champion(reason=reason)
            demoted = demote_result.get("demoted", False)
            out["demoted"] = demoted
            out["demote_reason"] = reason
            self.strikes = 0
        
        # Update shadow if requested
        if also_update_shadow and shadow_model_id:
            shadow_state = self.registry.update_shadow_day(
                shadow_model_id, date, ric, rsh, dd, at, self.cfg
            )
            out["shadow_status"] = shadow_state.status
            out["shadow_clean_days"] = shadow_state.clean_days
        
        return out


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing champion-challenger registry...")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ModelRegistry(tmpdir)
        
        # Create dummy model and metrics
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = (X["f1"] + np.random.randn(100) > 0).astype(int)
        dtr = lgb.Dataset(X, label=y)
        model = lgb.train({"objective": "binary", "verbose": -1}, dtr, num_boost_round=10)
        
        # Register first challenger
        m1 = ModelMetrics(ic_mean=0.02, ic_ir=0.5, book_sharpe=1.2, max_drawdown=0.08, 
                          avg_cost=0.0005, avg_turnover=0.8)
        mid1 = reg.register_challenger(model, "test_model_v1", m1, feature_cols=["f1", "f2"])
        print(f"Registered challenger: {mid1}")
        
        # Promote to champion (first model auto-promotes)
        gates = PromotionGate(min_book_sharpe=0.5)
        promo1 = reg.promote(mid1, "First model", gates=gates)
        print(f"Promotion 1: {promo1['promoted']}")
        
        # Register second challenger (better)
        m2 = ModelMetrics(ic_mean=0.03, ic_ir=0.7, book_sharpe=1.5, max_drawdown=0.06,
                          avg_cost=0.0004, avg_turnover=0.7)
        mid2 = reg.register_challenger(model, "test_model_v2", m2, feature_cols=["f1", "f2"])
        print(f"Registered challenger: {mid2}")
        
        # Compare
        cmp = reg.compare_to_champion(m2)
        print(f"Comparison: {cmp}")
        
        # Promote with gates (should pass - better Sharpe)
        gates2 = PromotionGate(min_book_sharpe=0.5)
        promo2 = reg.promote(mid2, "Better model", gates=gates2)
        print(f"Promotion 2: {promo2['promoted']}")
        
        # Check champion
        champ = reg.get_champion()
        print(f"Champion: {champ.model_id if champ else None}")
        
        # Test shadow evaluation
        print("\nTesting shadow evaluation...")
        reg.start_shadow(mid1, target_clean_days=5)
        
        kill_cfg = KillSwitchConfig(
            min_rolling_ic=-0.05,
            min_rolling_sharpe=-1.0,
            max_drawdown=0.15,
            max_avg_turnover=2.0,
            strikes_to_kill=3,
            shadow_clean_days=5,
        )
        
        monitor = KillSwitchMonitor(reg, kill_cfg)
        
        # Simulate clean days
        for i in range(10):
            st = monitor.update(
                pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                daily_ic=0.02,
                net_ret=0.001,
                turnover=0.5,
                also_update_shadow=True,
                shadow_model_id=mid1,
            )
            print(f"  Day {i}: clean={st.get('shadow_clean_days', 0)}, status={st.get('shadow_status', 'N/A')}")
        
        # Check shadow state
        is_clean, state = reg.is_shadow_clean(mid1)
        print(f"Shadow clean: {is_clean}, state: {state.status}, clean_days: {state.clean_days}")
        
        # Test kill-switch
        print("\nTesting kill-switch...")
        for i in range(5):
            st = monitor.update(
                pd.Timestamp("2023-01-15") + pd.Timedelta(days=i),
                daily_ic=-0.1,  # Bad IC
                net_ret=-0.01,  # Bad return
                turnover=2.5,   # High turnover
            )
            print(f"  Strike day {i}: strikes={st['strikes']}, breach={st['breach']}, demoted={st.get('demoted', False)}")
        
        # Check champion status
        champ = reg.get_champion()
        print(f"Champion after kill: {champ.model_id if champ else 'None (demoted)'}")
        
        print("\nAll tests passed!")
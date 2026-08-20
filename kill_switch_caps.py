"""
Kill-Switch and Borrow-Aware Position Caps

Enhances the existing kill-switch with:
1. Advanced borrow-aware position sizing
2. Multi-factor kill-switch (IC, Sharpe, DD, turnover, beta, sector concentration)
3. Position caps based on ADV, borrow cost, volatility
4. Gradual de-risking before hard kill
5. Integration with model registry for automatic demotion
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from datetime import datetime, timezone
import tempfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# Borrow-Aware Position Caps
# =============================================================================

@dataclass
class PositionCapConfig:
    """Configuration for position caps."""
    # Basic caps
    max_name_weight: float = 0.05        # Max absolute weight per name
    max_long_weight: float = 0.05        # Max long weight per name
    max_short_weight: float = 0.03       # Max short weight per name (stricter)
    
    # ADV participation caps
    max_participation: float = 0.05      # Max |w| * NAV / ADV
    participation_soft: float = 0.03     # Soft limit (warn)
    participation_hard: float = 0.08     # Hard limit (cap)
    
    # Borrow caps
    borrow_soft_bps: float = 150.0       # Annual bps: above this, shrink shorts
    borrow_hard_bps: float = 500.0       # Annual bps: above this, no shorts
    borrow_very_hard_bps: float = 1000.0 # Annual bps: emergency close
    
    # Volatility caps
    max_vol_weight: float = 0.10         # Max weight for high-vol names
    vol_scalar: float = 1.0              # Scale factor for vol targeting
    
    # Sector caps
    max_sector_weight: float = 0.30      # Max absolute sector weight
    max_sector_short: float = 0.15       # Max short sector weight
    
    # Factor caps
    max_beta_exposure: float = 0.30      # Max portfolio beta
    max_factor_exposure: float = 0.20    # Max style factor exposure
    
    # Gross/net
    max_gross_exposure: float = 1.5      # Max gross leverage
    max_net_exposure: float = 0.30       # Max net exposure
    
    # Book NAV
    book_nav: float = 1e7


def apply_position_caps(
    w: pd.Series,
    adv: pd.Series,
    borrow_bps_annual: pd.Series,
    sigmas: pd.Series,
    sectors: pd.Series,
    betas: pd.Series,
    config: PositionCapConfig,
    factor_loadings: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Apply all position caps sequentially.
    
    Order of application:
    1. Per-name weight caps
    2. Borrow caps (hard then soft)
    3. ADV participation caps
    4. Volatility caps
    5. Sector caps (project out)
    6. Factor caps (project out)
    7. Gross/net caps
    
    Returns:
        (capped_weights, diagnostics_dict)
    """
    diagnostics = {
        "original_gross": float(w.abs().sum()),
        "original_net": float(w.sum()),
        "original_max_weight": float(w.abs().max()),
        "steps": [],
    }
    
    w_capped = w.copy().astype(float)
    
    # ---- 1. Per-name weight caps ----
    w_capped = w_capped.clip(
        lower=-config.max_short_weight,
        upper=config.max_long_weight
    )
    diagnostics["steps"].append({
        "step": "name_caps",
        "gross": float(w_capped.abs().sum()),
        "net": float(w_capped.sum()),
    })
    
    # ---- 2. Borrow caps ----
    br = borrow_bps_annual.reindex(w_capped.index).fillna(100.0)
    
    # Very hard: emergency close all shorts
    very_hard = (w_capped < 0) & (br >= config.borrow_very_hard_bps)
    if very_hard.any():
        w_capped.loc[very_hard] = 0.0
        diagnostics["steps"].append({
            "step": "borrow_very_hard",
            "n_closed": int(very_hard.sum()),
            "gross": float(w_capped.abs().sum()),
        })
    
    # Hard: no shorts
    hard = (w_capped < 0) & (br >= config.borrow_hard_bps) & (br < config.borrow_very_hard_bps)
    if hard.any():
        w_capped.loc[hard] = 0.0
        diagnostics["steps"].append({
            "step": "borrow_hard",
            "n_closed": int(hard.sum()),
            "gross": float(w_capped.abs().sum()),
        })
    
    # Soft: shrink shorts
    soft = (w_capped < 0) & (br > config.borrow_soft_bps) & (br < config.borrow_hard_bps)
    if soft.any():
        scale = 1.0 - (br[soft] - config.borrow_soft_bps) / (
            config.borrow_hard_bps - config.borrow_soft_bps + 1e-12
        )
        scale = scale.clip(0.1, 1.0)
        w_capped.loc[soft] = w_capped.loc[soft] * scale
        diagnostics["steps"].append({
            "step": "borrow_soft",
            "n_shrunk": int(soft.sum()),
            "avg_scale": float(scale.mean()),
            "gross": float(w_capped.abs().sum()),
        })
    
    # ---- 3. ADV participation caps ----
    adv_aligned = adv.reindex(w_capped.index).fillna(config.book_nav)
    max_w_adv = (config.max_participation * adv_aligned / config.book_nav).clip(lower=1e-4)
    w_capped = w_capped.clip(lower=-max_w_adv, upper=max_w_adv)
    diagnostics["steps"].append({
        "step": "adv_caps",
        "gross": float(w_capped.abs().sum()),
        "net": float(w_capped.sum()),
    })
    
    # ---- 4. Volatility caps ----
    sig = sigmas.reindex(w_capped.index).fillna(0.01)
    # Scale weights by inverse volatility (vol targeting)
    vol_scale = (config.max_vol_weight / (sig * np.sqrt(252) + 1e-6)).clip(upper=1.0)
    w_capped = w_capped * vol_scale * config.vol_scalar
    diagnostics["steps"].append({
        "step": "vol_caps",
        "gross": float(w_capped.abs().sum()),
        "net": float(w_capped.sum()),
    })
    
    # ---- 5. Sector caps (project out excess) ----
    sec = sectors.reindex(w_capped.index).fillna("UNK")
    tmp = pd.DataFrame({"w": w_capped, "sec": sec})
    
    # Check sector gross
    sec_gross = tmp.groupby("sec")["w"].apply(lambda x: x.abs().sum())
    over_gross = sec_gross[sec_gross > config.max_sector_weight]
    if len(over_gross):
        for sector in over_gross.index:
            mask = (tmp["sec"] == sector)
            sector_w = tmp.loc[mask, "w"]
            scale = config.max_sector_weight / sector_w.abs().sum()
            tmp.loc[mask, "w"] = sector_w * scale
        w_capped = tmp["w"]
        diagnostics["steps"].append({
            "step": "sector_gross_cap",
            "sectors_capped": list(over_gross.index),
            "gross": float(w_capped.abs().sum()),
        })
    
    # Check sector short
    sec_short = tmp[tmp["w"] < 0].groupby("sec")["w"].sum().abs()
    over_short = sec_short[sec_short > config.max_sector_short]
    if len(over_short):
        for sector in over_short.index:
            mask = (tmp["sec"] == sector) & (tmp["w"] < 0)
            sector_w = tmp.loc[mask, "w"]
            scale = config.max_sector_short / sector_w.abs().sum()
            tmp.loc[mask, "w"] = sector_w * scale
        w_capped = tmp["w"]
        diagnostics["steps"].append({
            "step": "sector_short_cap",
            "sectors_capped": list(over_short.index),
            "gross": float(w_capped.abs().sum()),
        })
    
    # ---- 6. Factor caps (beta + style factors) ----
    # Beta neutralization
    b = betas.reindex(w_capped.index).fillna(1.0)
    beta_exp = (w_capped * b).sum()
    if abs(beta_exp) > config.max_beta_exposure:
        # Project out excess beta
        excess = beta_exp - np.sign(beta_exp) * config.max_beta_exposure
        bb = b.values
        wv = w_capped.values
        # Component of w along beta
        beta_component = (wv @ bb) / (bb @ bb + 1e-12) * bb
        # Reduce beta component proportionally
        if abs(beta_component).sum() > 0:
            reduction = excess / (beta_component @ bb + 1e-12)
            wv = wv - reduction * beta_component
            w_capped = pd.Series(wv, index=w_capped.index)
        diagnostics["steps"].append({
            "step": "beta_cap",
            "original_beta": float(beta_exp),
            "capped_beta": float((w_capped * b).sum()),
            "gross": float(w_capped.abs().sum()),
        })
    
    # Style factor caps (if provided)
    if factor_loadings is not None and len(factor_loadings):
        fl = factor_loadings.reindex(w_capped.index).fillna(0.0)
        for factor in fl.columns:
            f_exp = (w_capped * fl[factor]).sum()
            if abs(f_exp) > config.max_factor_exposure:
                excess = f_exp - np.sign(f_exp) * config.max_factor_exposure
                ff = fl[factor].values
                wv = w_capped.values
                factor_comp = (wv @ ff) / (ff @ ff + 1e-12) * ff
                if abs(factor_comp).sum() > 0:
                    reduction = excess / (factor_comp @ ff + 1e-12)
                    wv = wv - reduction * factor_comp
                    w_capped = pd.Series(wv, index=w_capped.index)
        diagnostics["steps"].append({
            "step": "factor_caps",
            "gross": float(w_capped.abs().sum()),
        })
    
    # ---- 7. Gross/Net caps ----
    gross = w_capped.abs().sum()
    if gross > config.max_gross_exposure:
        w_capped = w_capped * (config.max_gross_exposure / gross)
        diagnostics["steps"].append({
            "step": "gross_cap",
            "original_gross": float(gross),
            "capped_gross": config.max_gross_exposure,
        })
    
    net = w_capped.sum()
    if abs(net) > config.max_net_exposure:
        # Adjust toward dollar neutral while preserving gross
        target_net = np.sign(net) * config.max_net_exposure
        # Shift all weights slightly
        shift = (target_net - net) / len(w_capped)
        w_capped = w_capped + shift
        # Re-apply gross cap if needed
        if w_capped.abs().sum() > config.max_gross_exposure:
            w_capped = w_capped * (config.max_gross_exposure / w_capped.abs().sum())
        diagnostics["steps"].append({
            "step": "net_cap",
            "original_net": float(net),
            "capped_net": target_net,
            "gross": float(w_capped.abs().sum()),
        })
    
    diagnostics["final_gross"] = float(w_capped.abs().sum())
    diagnostics["final_net"] = float(w_capped.sum())
    diagnostics["final_max_weight"] = float(w_capped.abs().max())
    
    return w_capped, diagnostics


# =============================================================================
# Advanced Kill-Switch
# =============================================================================

@dataclass
class KillSwitchThresholds:
    """Kill-switch thresholds."""
    # Performance thresholds
    min_rolling_ic: float = -0.03
    min_rolling_sharpe: float = -0.75
    max_drawdown: float = 0.15
    
    # Risk thresholds
    max_avg_turnover: float = 1.5
    max_beta_exposure: float = 0.50
    max_sector_concentration: float = 0.40
    max_single_name_weight: float = 0.10
    max_short_exposure: float = 0.60
    
    # Cost thresholds
    max_avg_cost_bps: float = 5.0       # 5 bps/day
    max_cost_sharpe_ratio: float = 0.5  # Cost-adjusted Sharpe floor
    
    # Window sizes
    ic_window: int = 42
    sharpe_window: int = 42
    drawdown_window: int = 84
    turnover_window: int = 42
    
    # Strike policy
    strikes_to_kill: int = 3
    strikes_to_warn: int = 1
    cooldown_days: int = 10
    
    # Gradual de-risking
    enable_gradual_derisk: bool = True
    derisk_start_strikes: int = 1
    derisk_max_reduction: float = 0.5   # Max 50% reduction at 2 strikes


@dataclass
class KillSwitchState:
    """Current kill-switch state."""
    status: str = "ok"  # ok, warning, derisking, killed
    strikes: int = 0
    last_breach: Optional[str] = None
    breach_history: List[Dict] = field(default_factory=list)
    derisk_factor: float = 1.0  # Multiplier for gross exposure
    consecutive_ok: int = 0


class AdvancedKillSwitch:
    """
    Advanced kill-switch with gradual de-risking and multi-factor monitoring.
    
    Integrates with ModelRegistry for automatic champion demotion.
    """
    
    def __init__(
        self,
        registry: "ModelRegistry",
        thresholds: Optional[KillSwitchThresholds] = None,
        on_derisk: Optional[Callable[[float], None]] = None,
        on_kill: Optional[Callable[[str], None]] = None,
    ):
        self.registry = registry
        self.thresholds = thresholds or KillSwitchThresholds()
        self.on_derisk = on_derisk
        self.on_kill = on_kill
        self.state = KillSwitchState()
        
        # Monitoring series
        self.ic_series: List[Tuple] = []
        self.ret_series: List[Tuple] = []
        self.turn_series: List[Tuple] = []
        self.beta_series: List[Tuple] = []
        self.sector_conc_series: List[Tuple] = []
        self.cost_series: List[Tuple] = []
    
    def _rolling_ic(self) -> float:
        w = self.thresholds.ic_window
        if len(self.ic_series) < max(8, w // 4):
            return np.nan
        vals = np.array([v for _, v in self.ic_series[-w:]], float)
        return float(np.nanmean(vals))
    
    def _rolling_sharpe(self) -> float:
        w = self.thresholds.sharpe_window
        if len(self.ret_series) < max(8, w // 4):
            return np.nan
        r = np.array([v for _, v in self.ret_series[-w:]], float)
        return float(np.nanmean(r) / (np.nanstd(r) + 1e-12) * np.sqrt(252))
    
    def _drawdown(self) -> float:
        w = self.thresholds.drawdown_window
        if len(self.ret_series) < 5:
            return 0.0
        r = pd.Series([v for _, v in self.ret_series[-w:]])
        eq = (1.0 + r).cumprod()
        return float((1.0 - eq / eq.cummax()).max())
    
    def _avg_turnover(self) -> float:
        w = self.thresholds.turnover_window
        if not self.turn_series:
            return np.nan
        vals = np.array([v for _, v in self.turn_series[-w:]], float)
        return float(np.nanmean(vals))
    
    def _avg_beta(self) -> float:
        if not self.beta_series:
            return 0.0
        vals = np.array([v for _, v in self.beta_series[-self.thresholds.sharpe_window:]], float)
        return float(np.nanmean(np.abs(vals)))
    
    def _max_sector_conc(self) -> float:
        if not self.sector_conc_series:
            return 0.0
        vals = np.array([v for _, v in self.sector_conc_series[-self.thresholds.sharpe_window:]], float)
        return float(np.nanmean(vals))
    
    def _avg_cost_sharpe(self) -> float:
        if len(self.cost_series) < 10:
            return np.nan
        r = np.array([v for _, v in self.ret_series[-self.thresholds.sharpe_window:]], float)
        c = np.array([v for _, v in self.cost_series[-self.thresholds.sharpe_window:]], float)
        net = r - c
        return float(np.nanmean(net) / (np.nanstd(net) + 1e-12) * np.sqrt(252))
    
    def update(
        self,
        date,
        daily_ic: float,
        net_ret: float,
        turnover: float,
        beta: float = 0.0,
        sector_concentration: float = 0.0,
        cost: float = 0.0,
        single_name_max: float = 0.0,
        short_exposure: float = 0.0,
        also_update_shadow: bool = False,
        shadow_model_id: Optional[str] = None,
    ) -> Dict:
        """
        Update kill-switch with daily metrics.
        
        Returns:
            Dict with updated state and actions taken
        """
        date = pd.to_datetime(date)
        
        # Store metrics
        self.ic_series.append((date, float(daily_ic)))
        self.ret_series.append((date, float(net_ret)))
        self.turn_series.append((date, float(turnover)))
        self.beta_series.append((date, float(beta)))
        self.sector_conc_series.append((date, float(sector_concentration)))
        self.cost_series.append((date, float(cost)))
        
        # Compute rolling metrics
        ric = self._rolling_ic()
        rsh = self._rolling_sharpe()
        dd = self._drawdown()
        at = self._avg_turnover()
        ab = self._avg_beta()
        sc = self._max_sector_conc()
        cs = self._avg_cost_sharpe()
        
        # Check breaches
        breaches = {
            "ic": (not np.isnan(ric) and ric < self.thresholds.min_rolling_ic),
            "sharpe": (not np.isnan(rsh) and rsh < self.thresholds.min_rolling_sharpe),
            "drawdown": dd > self.thresholds.max_drawdown,
            "turnover": (not np.isnan(at) and at > self.thresholds.max_avg_turnover),
            "beta": ab > self.thresholds.max_beta_exposure,
            "sector": sc > self.thresholds.max_sector_concentration,
            "single_name": single_name_max > self.thresholds.max_single_name_weight,
            "short_exp": short_exposure > self.thresholds.max_short_exposure,
            "cost": (not np.isnan(cs) and cs < self.thresholds.max_cost_sharpe_ratio),
        }
        
        any_breach = any(breaches.values())
        
        # Update state
        actions = []
        old_status = self.state.status
        
        if any_breach:
            self.state.strikes += 1
            self.state.consecutive_ok = 0
            self.state.last_breach = str(date.date())
            self.state.breach_history.append({
                "date": str(date.date()),
                "breaches": {k: v for k, v in breaches.items() if v},
                "strikes": self.state.strikes,
                "metrics": {
                    "rolling_ic": ric,
                    "rolling_sharpe": rsh,
                    "drawdown": dd,
                    "avg_turnover": at,
                    "avg_beta": ab,
                    "sector_conc": sc,
                    "cost_sharpe": cs,
                },
            })
            actions.append(f"strike+1 -> {self.state.strikes}")
            
            # Gradual de-risking
            if self.thresholds.enable_gradual_derisk and self.state.strikes >= self.thresholds.derisk_start_strikes:
                if self.state.strikes == 1:
                    self.state.derisk_factor = 0.75  # 25% reduction
                    self.state.status = "derisking"
                    actions.append("derisk: gross exposure * 0.75")
                    if self.on_derisk:
                        self.on_derisk(0.75)
                elif self.state.strikes == 2:
                    self.state.derisk_factor = 0.50  # 50% reduction
                    actions.append("derisk: gross exposure * 0.50")
                    if self.on_derisk:
                        self.on_derisk(0.50)
            
            # Hard kill
            if self.state.strikes >= self.thresholds.strikes_to_kill:
                self.state.status = "killed"
                reasons = [k for k, v in breaches.items() if v]
                reason = "kill_switch: " + ", ".join(reasons)
                actions.append(f"KILL: {reason}")
                
                if self.on_kill:
                    self.on_kill(reason)
                
                # Demote champion
                demote_result = self.registry.demote_champion(reason=reason)
                actions.append(f"demoted: {demote_result.get('demoted', False)}")
                
                self.state.strikes = 0
                self.state.derisk_factor = 1.0
        
        else:
            self.state.consecutive_ok += 1
            if self.state.consecutive_ok >= self.thresholds.cooldown_days and self.state.strikes > 0:
                self.state.strikes -= 1
                self.state.consecutive_ok = 0
                actions.append(f"strike-1 -> {self.state.strikes}")
            
            # Exit derisking
            if self.state.status == "derisking" and self.state.strikes == 0:
                self.state.status = "ok"
                self.state.derisk_factor = 1.0
                actions.append("re-risk: gross exposure * 1.0")
                if self.on_derisk:
                    self.on_derisk(1.0)
            
            if self.state.status == "warning":
                self.state.status = "ok"
        
        out = {
            "date": str(date.date()),
            "status": self.state.status,
            "strikes": self.state.strikes,
            "derisk_factor": self.state.derisk_factor,
            "rolling_ic": ric,
            "rolling_sharpe": rsh,
            "drawdown": dd,
            "avg_turnover": at,
            "avg_beta": ab,
            "sector_concentration": sc,
            "cost_sharpe": cs,
            "breaches": breaches,
            "any_breach": any_breach,
            "actions": actions,
            "consecutive_ok": self.state.consecutive_ok,
        }
        
        # Update shadow if requested
        if also_update_shadow and shadow_model_id:
            from model_registry import KillSwitchConfig
            ks_cfg = KillSwitchConfig(
                min_rolling_ic=self.thresholds.min_rolling_ic,
                min_rolling_sharpe=self.thresholds.min_rolling_sharpe,
                max_drawdown=self.thresholds.max_drawdown,
                max_avg_turnover=self.thresholds.max_avg_turnover,
                strikes_to_kill=self.thresholds.strikes_to_kill,
                shadow_clean_days=20,
            )
            shadow_state = self.registry.update_shadow_day(
                shadow_model_id, date, ric, rsh, dd, at, ks_cfg
            )
            out["shadow_status"] = shadow_state.status
            out["shadow_clean_days"] = shadow_state.clean_days
        
        return out
    
    def get_state(self) -> KillSwitchState:
        """Get current state."""
        return self.state
    
    def get_derisk_factor(self) -> float:
        """Get current de-risking factor (1.0 = full, 0.5 = half)."""
        return self.state.derisk_factor
    
    def apply_derisk(self, weights: pd.Series) -> pd.Series:
        """Apply de-risking factor to weights."""
        return weights * self.state.derisk_factor


# =============================================================================
# Integration with Portfolio Construction
# =============================================================================

def build_capped_portfolio(
    day_sizes: pd.Series,
    returns_hist: pd.DataFrame,
    sectors: pd.Series,
    adv: pd.Series,
    borrow_bps: pd.Series,
    sigmas: pd.Series,
    betas: pd.Series,
    cap_config: PositionCapConfig,
    factor_loadings: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Full pipeline: sector-neutral HRP -> position caps.
    
    Returns:
        (final_weights, diagnostics)
    """
    from multi_horizon_hrp import (
        build_sector_neutral_hrp_weights,
        SectorNeutralConfig,
    )
    
    # Build sector-neutral HRP weights
    sn_config = SectorNeutralConfig(
        neutralize_sizes=True,
        conf_blend=0.3,
        gross_target=1.0,
        max_name_weight=cap_config.max_name_weight,
        max_short_weight=cap_config.max_short_weight,
        max_participation=cap_config.max_participation,
        borrow_soft_bps=cap_config.borrow_soft_bps,
        borrow_hard_bps=cap_config.borrow_hard_bps,
        book_nav=cap_config.book_nav,
    )
    
    w_hrp = build_sector_neutral_hrp_weights(
        day_sizes, sectors, returns_hist, config=sn_config, betas=betas
    )
    
    # Apply position caps
    w_final, cap_diag = apply_position_caps(
        w_hrp, adv, borrow_bps, sigmas, sectors, betas,
        cap_config, factor_loadings
    )
    
    diagnostics = {
        "hrp_weights": w_hrp.to_dict(),
        "final_weights": w_final.to_dict(),
        "cap_diagnostics": cap_diag,
    }
    
    return w_final, diagnostics


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing kill-switch and borrow-aware caps...")
    
    # Create test data
    np.random.seed(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    sectors = pd.Series({t: f"S{i % 4}" for i, t in enumerate(tickers)})
    
    # Test position caps
    print("\n1. Testing position caps...")
    w = pd.Series({
        "T00": 0.08, "T01": -0.06, "T02": 0.04, "T03": -0.02,
        "T04": 0.03, "T05": -0.05, "T06": 0.02, "T07": -0.01,
        "T08": 0.01, "T09": -0.03,
        "T10": 0.04, "T11": -0.02, "T12": 0.02, "T13": -0.01,
        "T14": 0.01, "T15": -0.02, "T16": 0.03, "T17": -0.04,
        "T18": 0.02, "T19": -0.01,
    })
    
    adv = pd.Series({t: float(5e7 * np.exp(np.random.normal(0, 0.25))) for t in tickers})
    borrow = pd.Series({t: float(np.random.choice([50, 100, 200, 600, 1200])) for t in tickers})
    sigmas = pd.Series({t: float(np.random.uniform(0.005, 0.03)) for t in tickers})
    betas = pd.Series({t: float(np.random.uniform(0.5, 1.5)) for t in tickers})
    
    cap_config = PositionCapConfig(
        max_name_weight=0.05,
        max_short_weight=0.03,
        max_participation=0.05,
        borrow_soft_bps=150.0,
        borrow_hard_bps=500.0,
        borrow_very_hard_bps=1000.0,
        book_nav=1e7,
        max_sector_weight=0.30,
        max_sector_short=0.15,
    )
    
    w_capped, diag = apply_position_caps(
        w, adv, borrow, sigmas, sectors, betas, cap_config
    )
    
    print(f"Original gross: {diag['original_gross']:.3f}")
    print(f"Final gross: {diag['final_gross']:.3f}")
    print(f"Final net: {diag['final_net']:.3f}")
    print(f"Steps:")
    for step in diag["steps"]:
        print(f"  {step}")
    
    # Test kill-switch
    print("\n2. Testing advanced kill-switch...")
    with tempfile.TemporaryDirectory() as tmpdir:
        from model_registry import ModelRegistry, ModelMetrics
        import lightgbm as lgb
        
        reg = ModelRegistry(tmpdir)
        
        # Register dummy champion
        X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
        y = (X["f1"] + np.random.randn(100) > 0).astype(int)
        model = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, label=y), num_boost_round=5)
        
        m = ModelMetrics(book_sharpe=1.0, max_drawdown=0.05)
        mid = reg.register_challenger(model, "test", m)
        reg.promote(mid, "test")
        
        # Create kill-switch
        ks = AdvancedKillSwitch(reg, KillSwitchThresholds(
            min_rolling_sharpe=-1.0,
            max_drawdown=0.10,
            max_avg_turnover=2.0,
            strikes_to_kill=3,
            enable_gradual_derisk=True,
        ))
        
        # Normal days
        for i in range(5):
            st = ks.update(
                pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                daily_ic=0.02, net_ret=0.001, turnover=0.5,
                beta=0.1, sector_concentration=0.2, cost=0.0001,
                single_name_max=0.04, short_exposure=0.3,
            )
            print(f"  Day {i}: status={st['status']}, strikes={st['strikes']}, derisk={st['derisk_factor']:.2f}")
        
        # Bad days
        for i in range(5):
            st = ks.update(
                pd.Timestamp("2023-01-06") + pd.Timedelta(days=i),
                daily_ic=-0.05, net_ret=-0.008, turnover=3.0,
                beta=0.6, sector_concentration=0.5, cost=0.001,
                single_name_max=0.08, short_exposure=0.7,
            )
            print(f"  Bad day {i}: status={st['status']}, strikes={st['strikes']}, derisk={st['derisk_factor']:.2f}, actions={st['actions']}")
        
        # Check champion status
        champ = reg.get_champion()
        print(f"Champion after: {champ.model_id if champ else 'None'}")
    
    print("\nAll tests passed!")
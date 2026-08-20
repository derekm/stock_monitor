"""
V5 Integrated Module - Complete End-to-End Pipeline

This module integrates all components into a single, coherent pipeline:
1. Data loading from Parquet feature store
2. Multi-horizon relevance labeling
3. Expanding-window cross-sectional LambdaRank
4. Conformal bet sizing
5. Sector-neutral HRP with borrow/ADV caps
6. Book backtest with costs/impact
7. Champion-challenger registry with shadow gate
8. Kill-switch monitoring
9. Artifact persistence

Usage:
    from v5_integrated import V5Pipeline, V5Config
    
    config = V5Config(...)
    pipeline = V5Pipeline(config)
    results = pipeline.run()
"""

from __future__ import annotations

import json
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Import all modules
from research_hygiene import PurgedKFold, frac_diff_ffd
from portfolio_construction import SectorNeutralConfig, hrp_weights_from_returns, estimate_betas, sector_neutralize_scores, project_weights_factor_neutral
from cross_sectional_ranker import (
    ExpandingRankerConfig, LambdaRankConfig,
    cs_relevance_from_y, add_multi_horizon_labels,
    train_lambdarank_ensemble, predict_ensemble,
    expanding_window_lambdarank as expanding_ranker,
)
from conformal_sizing import (
    ConformalState,
    expanding_conformal_sizes,
    fit_conformal,
    bet_size_from_conformal,
)
from expanding_backtest import (
    CostModel, BacktestConfig, build_book_backtest, apply_costs_and_impact, apply_borrow_cost,
)
from model_registry import (
    ModelRegistry, ModelMetadata, ModelMetrics, PromotionGate,
    KillSwitchConfig as RegistryKillSwitchConfig, KillSwitchMonitor,
)
from multi_horizon_hrp import (
    MultiHorizonConfig, LambdaRankConfig as MH_LambdaRankConfig,
    add_multi_horizon_labels as mh_add_labels,
    train_multih_lambdarank_ensemble, predict_multih_ensemble,
    expanding_multih_ranker, SectorNeutralConfig as MH_SectorConfig,
    sector_neutralize_scores as mh_sector_neutralize,
    project_weights_factor_neutral as mh_project_neutral,
    build_sector_neutral_hrp_weights,
)
from kill_switch_caps import (
    PositionCapConfig, KillSwitchThresholds, AdvancedKillSwitch,
    apply_position_caps, build_capped_portfolio,
)
from feature_store import ParquetFeatureStore, StoreConfig, create_store_from_data


# =============================================================================
# V5 Configuration
# =============================================================================

@dataclass
class V5Config:
    """Master configuration for V5 pipeline."""
    
    # Data
    # Parallelism. n_jobs=None -> cpu_count()-2 (leave headroom for the OS and
    # for LightGBM's own threads). lgb_num_threads bounds the ranker so it does
    # not oversubscribe the box when it runs alongside anything else.
    n_jobs: Optional[int] = None
    lgb_num_threads: Optional[int] = None

    store_root: str | Path = "./v5_feature_store"
    feature_cols: List[str] = field(default_factory=lambda: [
        "ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap",
    ])
    horizons: Tuple[int, ...] = (1, 5, 21)
    n_bins: int = 5
    
    # Expanding ranker
    min_train_dates: int = 252
    test_block: int = 21
    step: int = 21
    embargo_dates: int = 5
    valid_frac_of_train: float = 0.15
    
    # LightGBM
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31
    lgb_max_depth: int = 6
    lgb_num_boost_round: int = 300
    lgb_early_stopping: int = 40
    use_gpu: bool = False
    
    # Conformal
    conformal_alpha: float = 0.1
    conformal_calib_frac: float = 0.3
    conformal_min_calib: int = 100
    exp_conformal_halflife: int = 63
    
    # Portfolio construction
    conf_blend: float = 0.3
    gross_target: float = 1.0
    max_name_weight: float = 0.05
    max_short_weight: float = 0.03
    max_participation: float = 0.05
    borrow_soft_bps: float = 150.0
    borrow_hard_bps: float = 500.0
    book_nav: float = 1e7
    
    # Costs
    impact_coeff: float = 1e-6
    spread_bps: float = 5.0
    
    # Borrow
    borrow_cost_lookback: int = 1
    
    # Champion-challenger
    promotion_min_ic_mean: float = -0.02
    promotion_min_ic_ir: float = 0.2
    promotion_min_book_sharpe: float = 0.5
    promotion_max_drawdown: float = 0.15
    shadow_clean_days: int = 20
    
    # Kill-switch
    kill_min_rolling_ic: float = -0.03
    kill_min_rolling_sharpe: float = -0.75
    kill_max_drawdown: float = 0.15
    kill_max_avg_turnover: float = 1.5
    kill_strikes_to_kill: int = 3
    kill_enable_gradual: bool = True
    
    # Store
    store_partition_by_date: bool = True
    store_partition_freq: str = "M"
    store_compression: str = "snappy"
    
    # Run
    run_name: Optional[str] = None
    save_artifacts: bool = True
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"v5_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.store_root = Path(self.store_root)


# =============================================================================
# V5 Pipeline
# =============================================================================

def _book_summary(book: pd.DataFrame) -> Dict[str, float]:
    """Derive scalar book metrics from the per-DAY backtest frame.

    build_book_backtest emits one row per trading day with net_ret / gross_ret /
    cost / borrow_cost / turnover / book_beta. STEP 4 used to look for pre-aggregated
    columns instead ("net_sharpe", "max_drawdown", "cum_net", "avg_turnover"), which
    the backtest never produces -- so every lookup fell through to its `else np.nan`
    and the promotion gate was handed nan for Sharpe and drawdown. A gate can only
    reject on nan, never pass, so a strategy could not be promoted on book quality
    even in principle. The aggregation happens here instead.

    Sharpe is annualised from daily returns (252). Drawdown is on the compounded
    equity curve and returned POSITIVE (0.20 = a 20% peak-to-trough loss) because
    PromotionGate compares it against max_drawdown as a magnitude.
    """
    out = {
        "net_sharpe": float("nan"), "gross_sharpe": float("nan"),
        "avg_cost": float("nan"), "avg_borrow": float("nan"),
        "avg_turnover": float("nan"), "max_drawdown": float("nan"),
        "avg_book_beta": float("nan"), "cum_net": float("nan"),
    }
    if book is None or len(book) == 0:
        return out

    def _ann_sharpe(col: str) -> float:
        if col not in book.columns:
            return float("nan")
        r = pd.to_numeric(book[col], errors="coerce").dropna()
        if len(r) < 2:
            return float("nan")
        sd = r.std()
        if not np.isfinite(sd) or sd <= 0:
            return float("nan")
        return float(r.mean() / sd * np.sqrt(252.0))

    out["net_sharpe"] = _ann_sharpe("net_ret")
    out["gross_sharpe"] = _ann_sharpe("gross_ret")

    for key, col in (("avg_cost", "cost"), ("avg_borrow", "borrow_cost"),
                     ("avg_turnover", "turnover"), ("avg_book_beta", "book_beta")):
        if col in book.columns:
            v = pd.to_numeric(book[col], errors="coerce").mean()
            out[key] = float(v) if np.isfinite(v) else float("nan")

    if "net_ret" in book.columns:
        r = pd.to_numeric(book["net_ret"], errors="coerce").fillna(0.0)
        eq = (1.0 + r).cumprod()
        out["cum_net"] = float(eq.iloc[-1] - 1.0)
        peak = eq.cummax()
        dd = (eq / peak - 1.0).min()
        # positive magnitude: gates treat max_drawdown as "no worse than X"
        out["max_drawdown"] = float(-dd) if np.isfinite(dd) else float("nan")

    return out


def _build_one_date(task, sectors, adv, borrow, betas, cap_config):
    """Build one date's capped portfolio. Module level so it is picklable.

    Returns (weights, diagnostics) with weights.name set to the date string, or
    (None, None) when the date cannot be built. Exceptions are swallowed per date
    deliberately: one bad window must not abort a 2,000-date run, and the caller
    counts how many dates came back.
    """
    date_str, sizes, hist_returns = task
    try:
        sigmas = hist_returns.std() * np.sqrt(252)
        sigmas = sigmas.replace(0, np.nan).fillna(sigmas.median())
        w_final, diag = build_capped_portfolio(
            sizes, hist_returns, sectors, adv, borrow, sigmas, betas, cap_config,
        )
        w_final.name = date_str
        return w_final, diag
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {date_str}: {type(exc).__name__}: {exc}")
        return None, None

class V5Pipeline:
    """
    Complete V5 pipeline integrating all components.
    """
    
    def __init__(self, config: V5Config):
        self.config = config
        self.store = None
        self.registry = None
        self.kill_switch = None
        self.results = {}
        
        # Initialize store
        self.store = ParquetFeatureStore(StoreConfig(
            root=config.store_root,
            partition_by_date=config.store_partition_by_date,
            partition_freq=config.store_partition_freq,
            compression=config.store_compression,
        ))
        
        # Initialize registry
        self.registry = ModelRegistry(config.store_root / "registry")
        
        # Initialize kill-switch
        self.kill_switch = AdvancedKillSwitch(
            self.registry,
            KillSwitchThresholds(
                min_rolling_ic=config.kill_min_rolling_ic,
                min_rolling_sharpe=config.kill_min_rolling_sharpe,
                max_drawdown=config.kill_max_drawdown,
                max_avg_turnover=config.kill_max_avg_turnover,
                strikes_to_kill=config.kill_strikes_to_kill,
                enable_gradual_derisk=config.kill_enable_gradual,
            ),
        )
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load all data from feature store."""
        print("Loading data from feature store...")
        
        # Try to read existing panel
        panel = self.store.read_panel()
        returns_wide = self.store.read_returns_wide()
        sectors = self.store.read_static_map("sectors")
        adv = self.store.read_static_map("adv")
        borrow = self.store.read_static_map("borrow_bps_annual")
        
        if panel.empty or returns_wide.empty:
            raise ValueError("No data in feature store. Run create_store first.")
        
        print(f"  Panel: {panel.shape}, Returns: {returns_wide.shape}")
        print(f"  Sectors: {len(sectors)}, ADV: {len(adv)}, Borrow: {len(borrow)}")
        
        return panel, returns_wide, sectors, adv, borrow
    
    def run_ranking(self, panel: pd.DataFrame, returns_wide: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run expanding-window multi-horizon ranking."""
        print("\n" + "=" * 60)
        print("STEP 1: Multi-Horizon LambdaRank")
        print("=" * 60)
        
        # Add multi-horizon labels
        mh_config = MultiHorizonConfig(
            horizons=self.config.horizons,
            n_bins=self.config.n_bins,
        )
        
        panel = mh_add_labels(panel, returns_wide, mh_config)
        
        # Expanding window ranker
        exp_config = ExpandingRankerConfig(
            min_train_dates=self.config.min_train_dates,
            test_block=self.config.test_block,
            step=self.config.step,
            embargo_dates=self.config.embargo_dates,
            valid_frac_of_train=self.config.valid_frac_of_train,
            n_bins=self.config.n_bins,
            num_boost_round=self.config.lgb_num_boost_round,
            use_gpu=self.config.use_gpu,
        )
        
        ranker_config = MH_LambdaRankConfig(
            n_bins=self.config.n_bins,
            learning_rate=self.config.lgb_learning_rate,
            num_leaves=self.config.lgb_num_leaves,
            max_depth=self.config.lgb_max_depth,
            early_stopping_rounds=self.config.lgb_early_stopping,
            num_boost_round=self.config.lgb_num_boost_round,
            use_gpu=self.config.use_gpu,
        )
        
        oos_scores, window_stats, last_bundle = expanding_multih_ranker(
            panel, self.config.feature_cols, mh_config, exp_config, ranker_config
        )
        
        self.results["window_stats"] = window_stats
        self.results["last_bundle"] = {
            "ens_weights": last_bundle["ens_weights"],
            "horizons": last_bundle["horizons"],
            "horizon_weights": last_bundle["horizon_weights"],
        }
        
        if self.config.save_artifacts:
            self.store.write_artifact(oos_scores, self.config.run_name, "oos_scores")
        
        return oos_scores, last_bundle
    
    def run_conformal_sizing(self, oos_scores: pd.DataFrame) -> pd.DataFrame:
        """Run conformal bet sizing."""
        print("\n" + "=" * 60)
        print("STEP 2: Conformal Bet Sizing")
        print("=" * 60)
        
        # Expanding conformal for adaptive threshold
        sized = expanding_conformal_sizes(
            oos_scores, self.config.feature_cols,
            alpha_grid=(self.config.conformal_alpha,),
            min_train_dates=self.config.min_train_dates // 2,
            recal_every=self.config.step,
            embargo=self.config.embargo_dates,
        )
        
        self.results["conformal_summary"] = {
            "alpha": self.config.conformal_alpha,
            "n_sized": len(sized),
        }
        
        if self.config.save_artifacts:
            self.store.write_artifact(sized, self.config.run_name, "conformal_sizes")
        
        return sized
    
    def run_portfolio_construction(
        self,
        sized: pd.DataFrame,
        returns_wide: pd.DataFrame,
        sectors: pd.Series,
        adv: pd.Series,
        borrow: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run sector-neutral HRP with caps and book backtest."""
        print("\n" + "=" * 60)
        print("STEP 3: Portfolio Construction & Book Backtest")
        print("=" * 60)
        
        # Build capped portfolio per day
        cap_config = PositionCapConfig(
            max_name_weight=self.config.max_name_weight,
            max_short_weight=self.config.max_short_weight,
            max_participation=self.config.max_participation,
            borrow_soft_bps=self.config.borrow_soft_bps,
            borrow_hard_bps=self.config.borrow_hard_bps,
            book_nav=self.config.book_nav,
        )
        
        sn_config = MH_SectorConfig(
            neutralize_sizes=True,
            conf_blend=self.config.conf_blend,
            gross_target=self.config.gross_target,
            max_name_weight=self.config.max_name_weight,
            max_short_weight=self.config.max_short_weight,
            max_participation=self.config.max_participation,
            borrow_soft_bps=self.config.borrow_soft_bps,
            borrow_hard_bps=self.config.borrow_hard_bps,
            book_nav=self.config.book_nav,
        )
        
        # Estimate betas
        betas = estimate_betas(returns_wide)
        
        # Per-day portfolio construction
        all_weights = []
        all_diagnostics = []

        dates = sorted(sized["date"].unique())

        # Each date is independent: it reads its own trailing 120-day window and
        # produces one weight vector, so the loop is embarrassingly parallel. It is
        # also the dominant cost of this step (one HRP per long/short sleeve per
        # date), which is why this is parallelised and the ranker is not.
        #
        # Processes, not threads: the work is numpy/scipy/pandas under the GIL, and
        # scipy.linkage is single-threaded CPU. Payload per task is only the day's
        # sizes plus a 120-row window slice.
        #
        # GPU note: _cov_corr routes to CUDA above _CUDA_MIN_NAMES names. A 10k x 10k
        # fp32 matrix is 0.40GB against ~1.65GB free on this card, so N workers each
        # holding one would OOM. Workers are therefore capped so that at most a few
        # can be resident, and each worker frees its device memory per call.
        workers = self.config.n_jobs
        if workers in (None, 0):
            workers = max(1, (os.cpu_count() or 2) - 2)
        workers = max(1, min(workers, len(dates)))

        tasks = []
        for date in dates:
            day_sized = sized[sized["date"] == date].set_index("ticker")
            hist_returns = returns_wide[returns_wide.index < date].tail(120)
            if len(hist_returns) < 20:
                continue
            tasks.append((str(date), day_sized["size_raw"], hist_returns))

        if workers == 1 or len(tasks) <= 1:
            results = [
                _build_one_date(t, sectors, adv, borrow, betas, cap_config)
                for t in tasks
            ]
        else:
            print(f"  building {len(tasks)} dates across {workers} processes...")
            results = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(
                        _build_one_date, t, sectors, adv, borrow, betas, cap_config
                    ): i
                    for i, t in enumerate(tasks)
                }
                done = 0
                for fut in as_completed(futs):
                    i = futs[fut]
                    results[i] = fut.result()
                    done += 1
                    if done % 200 == 0 or done == len(tasks):
                        print(f"    {done}/{len(tasks)} dates")

        for w_final, diag in results:
            if w_final is None:
                continue
            all_weights.append(w_final)
            all_diagnostics.append(diag)
        
        weights_df = pd.DataFrame(all_weights).T if all_weights else pd.DataFrame()
        weights_df.index.name = "ticker"
        
        # Reshape weights_df for book backtest (needs date, ticker, size_raw columns)
        # weights_df: index=ticker, columns=dates
        # We need to melt it to long format
        sized_for_bt = weights_df.reset_index().melt(
            id_vars="ticker", var_name="date", value_name="size_raw"
        )
        sized_for_bt = sized_for_bt[sized_for_bt["size_raw"] != 0].dropna()
        
        # Book backtest
        bt_config = BacktestConfig(
            lookback=120,
            gross_target=self.config.gross_target,
            cost=CostModel(
                impact_coeff=self.config.impact_coeff,
                half_spread_bps=self.config.spread_bps,
            ),
            sector_neutral=True,
            conf_blend=self.config.conf_blend,
        )
        book_results = build_book_backtest(
            sized_for_bt, returns_wide,
            sectors=sectors, adv=adv, borrow_bps_annual=borrow,
            config=bt_config,
        )
        diag = {}
        
        self.results["book_backtest"] = book_results
        self.results["portfolio_diagnostics"] = all_diagnostics
        
        if self.config.save_artifacts:
            self.store.write_artifact(weights_df, self.config.run_name, "daily_weights")
            self.store.write_artifact(book_results, self.config.run_name, "book_backtest")
        
        return weights_df, book_results, diag
    
    def register_and_evaluate(
        self,
        oos_scores: pd.DataFrame,
        book_results: pd.DataFrame,
        last_bundle: Dict,
    ) -> Tuple[str, bool]:
        """Register model and evaluate for promotion."""
        print("\n" + "=" * 60)
        print("STEP 4: Champion-Challenger Evaluation")
        print("=" * 60)
        
        # Compute metrics
        ic = oos_scores.groupby("date").apply(
            lambda g: g["score"].corr(g["y"])
            if g["y"].nunique() > 1 and g["score"].nunique() > 1 else np.nan
        )

        book = _book_summary(book_results)

        metrics = ModelMetrics(
            ic_mean=float(np.nanmean(ic)),
            ic_ir=float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)),
            book_sharpe=book["net_sharpe"],
            gross_sharpe=book["gross_sharpe"],
            avg_cost=book["avg_cost"],
            avg_borrow=book["avg_borrow"],
            avg_turnover=book["avg_turnover"],
            max_drawdown=book["max_drawdown"],
            avg_book_beta=book["avg_book_beta"],
            cum_net=book["cum_net"],
            n_oos_rows=len(oos_scores),
            n_test_days=oos_scores["date"].nunique(),
        )
        
        # Register challenger
        model_id = self.registry.register_challenger(
            last_bundle["models"]["blend"],  # Use blended model
            "v5_pipeline",
            metrics,
            params=asdict(self.config),
            feature_cols=self.config.feature_cols,
            extra={
                "horizons": self.config.horizons,
                "ens_weights": last_bundle["ens_weights"],
            },
        )
        
        print(f"Registered challenger: {model_id}")
        print(f"Metrics: IC={metrics.ic_mean:.4f}, IR={metrics.ic_ir:.2f}, "
              f"Book Sharpe={metrics.book_sharpe:.2f}, DD={metrics.max_drawdown:.2%}")
        
        # Promotion gates
        gates = PromotionGate(
            min_ic_mean=self.config.promotion_min_ic_mean,
            min_ic_ir=self.config.promotion_min_ic_ir,
            min_book_sharpe=self.config.promotion_min_book_sharpe,
            max_drawdown=self.config.promotion_max_drawdown,
            require_shadow_clean_days=self.config.shadow_clean_days,
        )
        
        promo_result = self.registry.promote(model_id, "V5 pipeline run", gates=gates)
        
        if promo_result["promoted"]:
            print(f"PROMOTED to champion!")
        else:
            print(f"NOT promoted: {promo_result['reason']}")
        
        # Start shadow evaluation
        self.registry.start_shadow(model_id, target_clean_days=self.config.shadow_clean_days)
        
        return model_id, promo_result["promoted"]
    
    def run_kill_switch_check(
        self,
        book_results: pd.DataFrame,
        champion_model_id: Optional[str] = None,
    ) -> Dict:
        """Run kill-switch monitoring on book backtest."""
        print("\n" + "=" * 60)
        print("STEP 5: Kill-Switch Monitoring")
        print("=" * 60)
        
        # Replay daily metrics through kill-switch
        for idx, row in book_results.iterrows():
            st = self.kill_switch.update(
                idx,
                daily_ic=row.get("daily_ic", np.nan),
                net_ret=row.get("net_ret", 0.0),
                turnover=row.get("turnover", 0.0),
                beta=row.get("book_beta", 0.0),
                sector_concentration=row.get("max_sector_conc", 0.0),
                cost=row.get("cost", 0.0),
                single_name_max=row.get("max_weight", 0.0),
                short_exposure=row.get("short_exposure", 0.0),
                also_update_shadow=True,
                shadow_model_id=champion_model_id,
            )
            
            if st["status"] in ("derisking", "killed"):
                print(f"  {st['date']}: {st['status']} - {st['actions']}")
        
        final_state = self.kill_switch.get_state()
        print(f"Final state: {final_state.status}, strikes: {final_state.strikes}, "
              f"derisk_factor: {final_state.derisk_factor:.2f}")
        
        self.results["kill_switch_state"] = asdict(final_state)
        
        return asdict(final_state)
    
    def save_summary(self) -> Dict:
        """Save run summary."""
        summary = {
            "run_name": self.config.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "results": self.results,
        }
        
        if self.config.save_artifacts:
            self.store.write_run_summary(self.config.run_name, summary)
        
        return summary
    
    def run(self, panel: Optional[pd.DataFrame] = None, returns_wide: Optional[pd.DataFrame] = None,
            sectors: Optional[pd.Series] = None, adv: Optional[pd.Series] = None,
            borrow: Optional[pd.Series] = None) -> Dict:
        """Run complete V5 pipeline."""
        print("\n" + "=" * 60)
        print(f"V5 PIPELINE: {self.config.run_name}")
        print("=" * 60)
        
        # Load data if not provided
        if panel is None:
            panel, returns_wide, sectors, adv, borrow = self.load_data()
        
        # Step 1: Ranking
        oos_scores, last_bundle = self.run_ranking(panel, returns_wide)
        
        # Step 2: Conformal sizing
        sized = self.run_conformal_sizing(oos_scores)
        
        # Step 3: Portfolio construction
        weights_df, book_results, diag = self.run_portfolio_construction(
            sized, returns_wide, sectors, adv, borrow
        )
        
        # Step 4: Champion-challenger
        model_id, promoted = self.register_and_evaluate(oos_scores, book_results, last_bundle)
        
        # Step 5: Kill-switch
        ks_state = self.run_kill_switch_check(book_results, model_id if promoted else None)
        
        # Save summary
        summary = self.save_summary()
        
        print("\n" + "=" * 60)
        print("V5 PIPELINE COMPLETE")
        print("=" * 60)
        
        return summary


# =============================================================================
# Convenience: Create Store from Raw Data
# =============================================================================

def create_v5_store_from_raw(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    sectors: pd.Series,
    adv: pd.Series,
    borrow: pd.Series,
    store_root: str | Path,
    feature_cols: List[str],
    run_name: Optional[str] = None,
) -> ParquetFeatureStore:
    """Create V5 feature store from raw data."""
    return create_store_from_data(
        panel, returns_wide, sectors, adv, borrow,
        store_root, feature_cols, run_name
    )


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("Testing V5 Integrated Pipeline...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store_root = Path(tmpdir) / "v5_store"
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-01", periods=600)
        tickers = [f"T{i:02d}" for i in range(15)]
        
        # Returns
        rets = {}
        for tkr in tickers:
            rets[tkr] = np.random.randn(len(dates)) * 0.01
        returns_wide = pd.DataFrame(rets, index=dates)
        
        # Panel with features
        panels = []
        for tkr in tickers:
            close = 100 * np.exp(np.cumsum(rets[tkr]))
            for i, dt in enumerate(dates):
                if i < 20:
                    continue
                r1 = close[i] / close[i-1] - 1
                r5 = close[i] / close[i-5] - 1
                r10 = close[i] / close[i-10] - 1
                v10 = pd.Series(close[max(0,i-20):i+1]).pct_change().std()
                v20 = pd.Series(close[max(0,i-40):i+1]).pct_change().std()
                ma20 = close[max(0,i-20):i+1].mean()
                ma_gap = close[i] / ma20 - 1
                
                panels.append({
                    "date": dt,
                    "ticker": tkr,
                    "ret_1": r1,
                    "ret_5": r5,
                    "ret_10": r10,
                    "vol_10": v10,
                    "vol_20": v20,
                    "ma_gap": ma_gap,
                })
        
        panel = pd.DataFrame(panels)
        feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
        
        # Static maps
        sectors = pd.Series({t: f"S{i % 3}" for i, t in enumerate(tickers)}, name="sector")
        adv = pd.Series({t: float(5e7 * np.exp(np.random.normal(0, 0.25))) for t in tickers}, name="adv")
        borrow = pd.Series({t: float(np.random.choice([50, 100, 200, 500])) for t in tickers}, name="borrow_bps_annual")
        
        # Create store
        print("Creating feature store...")
        store = create_v5_store_from_raw(
            panel, returns_wide, sectors, adv, borrow,
            store_root, feature_cols, "test_v5"
        )
        
        # Run V5 pipeline
        config = V5Config(
            store_root=store_root,
            feature_cols=feature_cols,
            min_train_dates=100,
            test_block=21,
            step=21,
            embargo_dates=5,
            lgb_num_boost_round=50,
            run_name="test_v5",
            save_artifacts=True,
        )
        
        pipeline = V5Pipeline(config)
        summary = pipeline.run(panel, returns_wide, sectors, adv, borrow)
        
        print(f"\nRun summary saved: {config.run_name}")
        print(f"Champion promoted: {summary['results'].get('model_id') is not None}")
        
    print("\nAll tests passed!")
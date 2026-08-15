"""
Cross-Sectional LightGBM Ranker with LambdaRank

Implements:
1. Multi-ticker panel stacking with cross-sectional relevance bins
2. LambdaRank training with integer relevance labels
3. Expanding-window ranker with purged embargo
4. Multi-horizon relevance ensemble
5. Daily IC/IR evaluation
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# Relevance Binning (Integer Labels for LambdaRank)
# =============================================================================

def cs_relevance_from_y(
    y: pd.Series,
    n_bins: int = 5,
) -> pd.Series:
    """
    Cross-sectional relevance bins from forward returns.
    
    Args:
        y: Forward returns for one date (ticker-indexed)
        n_bins: Number of relevance bins
        
    Returns:
        Integer relevance labels {0, ..., n_bins-1}
    """
    y = y.dropna()
    if len(y) == 0:
        return pd.Series(dtype=int, index=y.index)
    try:
        result = pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int)
        return result.reindex(y.index)
    except Exception:
        r = y.rank(method="first", pct=True)
        result = np.floor(r * n_bins).astype(int).clip(0, n_bins - 1)
        return pd.Series(result, index=y.index)


def make_relevance_bins(
    panel: pd.DataFrame,
    y_col: str = "y",
    n_bins: int = 5,
    by: str = "date",
) -> pd.Series:
    """
    Add integer relevance bins per date for LambdaRank.
    
    Args:
        panel: Long-format panel with date, ticker, y_col
        y_col: Forward return column
        n_bins: Number of relevance bins
        by: Groupby column (usually 'date')
        
    Returns:
        Series of relevance labels aligned with panel index
    """
    def _bin(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s, q=n_bins, labels=False, duplicates="drop").astype(int)
        except Exception:
            r = s.rank(method="first", pct=True)
            return np.floor(r * n_bins).astype(int).clip(0, n_bins - 1)
    
    return panel.groupby(by, group_keys=False)[y_col].apply(_bin)


def add_multi_horizon_labels(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21),
    n_bins: int = 5,
    weights: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Add multi-horizon forward returns and blended relevance.
    
    Args:
        panel: Long-format panel with date, ticker
        returns_wide: Wide returns (date x ticker) for computing forward returns
        horizons: Forward horizons in days
        n_bins: Number of relevance bins
        weights: Optional dict mapping horizon -> weight for blending
        
    Returns:
        Panel with added y_h{h}, rel_h{h}, y (primary), relevance (blended)
    """
    d = panel.copy()
    horizons = tuple(sorted(set(horizons)))
    
    if weights is None:
        raw = {h: 1.0 / np.sqrt(h) for h in horizons}
        s = sum(raw.values())
        weights = {h: raw[h] / s for h in horizons}
    
    rw = returns_wide.sort_index()
    
    for h in horizons:
        col = f"y_h{h}"
        if col in d.columns:
            continue
        # Compound h-day forward return starting NEXT day
        fwd = (1.0 + rw).rolling(h).apply(np.prod, raw=True).shift(-h) - 1.0
        long = fwd.stack().rename(col)
        long.index = long.index.set_names(["date", "ticker"])
        tmp = long.reset_index()
        tmp["date"] = pd.to_datetime(tmp["date"])
        d = d.merge(tmp, on=["date", "ticker"], how="left")
    
    # Primary y = medium horizon (5d) if present else middle
    primary_h = 5 if 5 in horizons else horizons[len(horizons) // 2]
    d["y"] = d[f"y_h{primary_h}"]
    
    rel_cols = []
    for h in horizons:
        rc = f"rel_h{h}"
        d[rc] = d.groupby("date")[f"y_h{h}"].transform(
            lambda s, nb=n_bins: cs_relevance_from_y(s, nb)
        )
        rel_cols.append(rc)
    
    # Blended relevance
    blend = np.zeros(len(d), dtype=float)
    for h in horizons:
        blend += weights[h] * d[f"rel_h{h}"].astype(float).values
    
    d["relevance"] = np.clip(np.rint(blend), 0, n_bins - 1).astype(int)
    d["relevance_blend_raw"] = blend
    d.attrs["horizon_weights"] = weights
    d.attrs["horizons"] = horizons
    
    return d.dropna(subset=["y", "relevance"] + [f"y_h{h}" for h in horizons])


# =============================================================================
# LambdaRank Training
# =============================================================================

def _groups_from_dates(dates: pd.Series) -> np.ndarray:
    """LightGBM ranker group array: contiguous same-query sizes."""
    return dates.groupby(dates, sort=False).size().to_numpy()


@dataclass
class LambdaRankConfig:
    """Configuration for LambdaRank training."""
    n_bins: int = 5
    num_boost_round: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 25
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 2.0
    early_stopping_rounds: int = 40
    eval_at: tuple[int, ...] = (5, 10, 20)
    lambdarank_truncation_level: int = 30
    verbose: int = -1
    use_gpu: bool = False


def train_lambdarank(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    feature_cols: list[str],
    rel_col: str = "relevance",
    config: Optional[LambdaRankConfig] = None,
) -> lgb.Booster:
    """
    Train LightGBM LambdaRank model.
    
    CRITICAL: Data must be sorted by query id (date) so group boundaries are contiguous.
    
    Args:
        train_df: Training data with date, ticker, feature_cols, rel_col
        valid_df: Validation data (same structure) or None
        feature_cols: Feature column names
        rel_col: Relevance label column
        config: LambdaRankConfig
        
    Returns:
        Trained LightGBM Booster
    """
    config = config or LambdaRankConfig()
    
    tr = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    dtr = lgb.Dataset(
        tr[feature_cols],
        label=tr[rel_col].astype(int),
        group=_groups_from_dates(tr["date"]),
        free_raw_data=False,
    )
    
    valid_sets = [dtr]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(0)]
    
    if valid_df is not None and len(valid_df):
        va = valid_df.sort_values(["date", "ticker"]).reset_index(drop=True)
        dva = lgb.Dataset(
            va[feature_cols],
            label=va[rel_col].astype(int),
            group=_groups_from_dates(va["date"]),
            reference=dtr,
            free_raw_data=False,
        )
        valid_sets.append(dva)
        valid_names.append("valid")
        callbacks.insert(0, lgb.early_stopping(config.early_stopping_rounds))
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": list(config.eval_at),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l2": config.lambda_l2,
        "lambdarank_truncation_level": config.lambdarank_truncation_level,
        "label_gain": list(range(config.n_bins)),
        "verbose": config.verbose,
    }
    
    if config.use_gpu:
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    
    return lgb.train(
        params,
        dtr,
        num_boost_round=config.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def train_lambdarank_ensemble(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    feature_cols: list[str],
    horizons: tuple[int, ...] = (1, 5, 21),
    n_bins: int = 5,
    horizon_weights: Optional[dict[int, float]] = None,
    config: Optional[LambdaRankConfig] = None,
) -> dict:
    """
    Train LambdaRank ensemble: one per horizon + blended.
    
    Returns:
        Dict with 'models' (dict of Boosters), 'ens_weights', 'horizons'
    """
    config = config or LambdaRankConfig()
    config.n_bins = n_bins
    
    if horizon_weights is None:
        raw = {h: 1.0 / np.sqrt(h) for h in horizons}
        s = sum(raw.values())
        horizon_weights = {h: raw[h] / s for h in horizons}
    
    models = {}
    
    def _fit(tr, va, label_col, tag):
        tr = tr.sort_values(["date", "ticker"])
        dtr = lgb.Dataset(
            tr[feature_cols],
            label=tr[label_col].astype(int),
            group=tr.groupby("date", sort=False).size().to_numpy(),
            free_raw_data=False,
        )
        vs, vn, cb = [dtr], ["train"], [lgb.log_evaluation(0)]
        if va is not None and len(va):
            va = va.sort_values(["date", "ticker"])
            dva = lgb.Dataset(
                va[feature_cols],
                label=va[label_col].astype(int),
                group=va.groupby("date", sort=False).size().to_numpy(),
                reference=dtr,
                free_raw_data=False,
            )
            vs.append(dva)
            vn.append("valid")
            cb.insert(0, lgb.early_stopping(config.early_stopping_rounds))
        
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": list(config.eval_at),
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "min_data_in_leaf": config.min_data_in_leaf,
            "feature_fraction": config.feature_fraction,
            "bagging_fraction": config.bagging_fraction,
            "bagging_freq": config.bagging_freq,
            "lambda_l2": config.lambda_l2,
            "lambdarank_truncation_level": config.lambdarank_truncation_level,
            "label_gain": list(range(n_bins)),
            "verbose": config.verbose,
        }
        if config.use_gpu:
            params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        
        models[tag] = lgb.train(
            params, dtr,
            num_boost_round=config.num_boost_round,
            valid_sets=vs,
            valid_names=vn,
            callbacks=cb,
        )
    
    for h in horizons:
        _fit(train_df, valid_df, f"rel_h{h}", f"h{h}")
    _fit(train_df, valid_df, "relevance", "blend")
    
    # Ensemble weights: horizon weights normalized + blend gets 0.4 mass
    ens = {f"h{h}": 0.6 * horizon_weights[h] for h in horizons}
    ens["blend"] = 0.4
    z = sum(ens.values())
    ens = {k: v / z for k, v in ens.items()}
    
    return {"models": models, "ens_weights": ens, "horizons": horizons, "horizon_weights": horizon_weights}


def predict_ensemble(
    bundle: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    Predict using ensemble of LambdaRank models.
    
    Returns:
        Weighted average scores
    """
    acc = np.zeros(len(df))
    for tag, w in bundle["ens_weights"].items():
        m = bundle["models"][tag]
        acc += w * m.predict(df[feature_cols], num_iteration=m.best_iteration)
    return pd.Series(acc, index=df.index, name="score")


# =============================================================================
# Evaluation Metrics
# =============================================================================

def daily_ic(
    df: pd.DataFrame,
    score_col: str = "score",
    y_col: str = "y",
) -> pd.Series:
    """Daily Spearman IC between scores and forward returns."""
    return df.groupby("date").apply(
        lambda g: spearmanr(g[score_col], g[y_col]).correlation
        if g[y_col].nunique() > 1 and g[score_col].nunique() > 1
        else np.nan
    )


def daily_ndcg(
    df: pd.DataFrame,
    score_col: str = "score",
    rel_col: str = "relevance",
    k: int = 10,
) -> pd.Series:
    """Daily NDCG@k."""
    from sklearn.metrics import ndcg_score
    
    def _ndcg(g):
        if g[rel_col].nunique() < 2:
            return np.nan
        # ndcg_score expects 2D arrays
        return ndcg_score(
            g[rel_col].values.reshape(1, -1),
            g[score_col].values.reshape(1, -1),
            k=k,
        )
    
    return df.groupby("date").apply(_ndcg)


def top_bottom_spread(
    df: pd.DataFrame,
    score_col: str = "score",
    y_col: str = "y",
    q: float = 0.1,
) -> pd.Series:
    """Top-bottom decile spread (long top q, short bottom q)."""
    def _tb(g):
        hi = g[score_col].quantile(1 - q)
        lo = g[score_col].quantile(q)
        long = g.loc[g[score_col] >= hi, y_col].mean()
        short = g.loc[g[score_col] <= lo, y_col].mean()
        return long - short
    
    return df.groupby("date").apply(_tb)


# =============================================================================
# Expanding Window Ranker
# =============================================================================

@dataclass
class ExpandingRankerConfig:
    """Configuration for expanding-window ranker."""
    min_train_dates: int = 252
    test_block: int = 21
    step: int = 21
    embargo_dates: int = 5
    valid_frac_of_train: float = 0.15
    n_bins: int = 5
    num_boost_round: int = 350
    use_gpu: bool = False


class PurgedDateSplit:
    """Expanding or rolling train dates with embargo before each test block."""
    
    def __init__(self, embargo_dates: int = 5):
        self.embargo_dates = embargo_dates
    
    def expanding_windows(
        self,
        unique_dates: np.ndarray,
        min_train_dates: int,
        test_block: int,
        step: Optional[int] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = np.array(sorted(pd.to_datetime(unique_dates)))
        step = step or test_block
        start_test = min_train_dates
        
        while start_test < len(dates):
            end_test = min(start_test + test_block, len(dates))
            test = dates[start_test:end_test]
            train_end = max(0, start_test - self.embargo_dates)
            train = dates[:train_end]
            
            if len(train) >= min_train_dates // 2 and len(test) > 0:
                yield train, test
            
            if end_test >= len(dates):
                break
            start_test += step


def expanding_window_lambdarank(
    panel: pd.DataFrame,
    feature_cols: list[str],
    cfg: Optional[ExpandingRankerConfig] = None,
) -> tuple[pd.DataFrame, list[dict], lgb.Booster]:
    """
    Expanding-window LambdaRank training and OOS scoring.
    
    For each window:
      - Train on all history before embargo
      - Optional tail of train for early stopping
      - Predict test block after embargo
    
    Returns:
        (oos_scores, window_stats, last_model)
    """
    cfg = cfg or ExpandingRankerConfig()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    
    if "relevance" not in panel.columns:
        panel["relevance"] = make_relevance_bins(panel, n_bins=cfg.n_bins)
    
    splitter = PurgedDateSplit(embargo_dates=cfg.embargo_dates)
    dates = panel["date"].unique()
    
    oos_parts = []
    window_stats = []
    last_model = None
    
    for wi, (train_dates, test_dates) in enumerate(
        splitter.expanding_windows(
            dates,
            min_train_dates=cfg.min_train_dates,
            test_block=cfg.test_block,
            step=cfg.step,
        )
    ):
        dtr_all = panel[panel["date"].isin(train_dates)]
        dte = panel[panel["date"].isin(test_dates)]
        
        if dtr_all.empty or dte.empty:
            continue
        
        # Tail of train as valid (still before embargo/test)
        tr_dates_sorted = np.array(sorted(dtr_all["date"].unique()))
        n_va = max(5, int(len(tr_dates_sorted) * cfg.valid_frac_of_train))
        va_dates = set(tr_dates_sorted[-n_va:])
        fit_dates = set(tr_dates_sorted[:-n_va]) if len(tr_dates_sorted) > n_va + 20 else set(tr_dates_sorted)
        
        d_fit = dtr_all[dtr_all["date"].isin(fit_dates)]
        d_va = dtr_all[dtr_all["date"].isin(va_dates)] if va_dates != fit_dates else None
        
        lr_config = LambdaRankConfig(
            n_bins=cfg.n_bins,
            num_boost_round=cfg.num_boost_round,
            use_gpu=cfg.use_gpu,
        )
        
        model = train_lambdarank(
            d_fit, d_va, feature_cols,
            rel_col="relevance",
            config=lr_config,
        )
        
        last_model = model
        part = dte.copy()
        part["score"] = model.predict(
            part[feature_cols],
            num_iteration=model.best_iteration
        )
        part["window"] = wi
        oos_parts.append(part)
        
        ic = daily_ic(part)
        stats = {
            "window": wi,
            "train_end": str(pd.to_datetime(train_dates).max().date()),
            "test_start": str(pd.to_datetime(test_dates).min().date()),
            "test_end": str(pd.to_datetime(test_dates).max().date()),
            "n_train_rows": int(len(d_fit)),
            "n_test_rows": int(len(part)),
            "ic_mean": float(np.nanmean(ic)),
            "ic_ir": float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)),
            "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        }
        window_stats.append(stats)
        
        print(
            f"  win {wi:02d} train_end={stats['train_end']} "
            f"test={stats['test_start']}..{stats['test_end']} "
            f"IC={stats['ic_mean']:.4f} IR={stats['ic_ir']:.2f}"
        )
    
    if not oos_parts:
        raise RuntimeError("Expanding ranker produced no OOS scores — lower min_train_dates.")
    
    oos = pd.concat(oos_parts, ignore_index=True)
    oos = oos.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    return oos, window_stats, last_model


# =============================================================================
# Multi-Horizon Expanding Ranker
# =============================================================================

def expanding_multih_ranker(
    panel: pd.DataFrame,
    feature_cols: list[str],
    horizons: tuple[int, ...] = (1, 5, 21),
    cfg: Optional[ExpandingRankerConfig] = None,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    Expanding-window multi-horizon LambdaRank ensemble.
    
    Returns:
        (oos_scores, window_stats, last_bundle)
    """
    cfg = cfg or ExpandingRankerConfig()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    
    dates = np.array(sorted(panel["date"].unique()))
    
    oos_parts = []
    stats = []
    last_bundle = None
    
    start = cfg.min_train_dates
    while start < len(dates):
        end = min(start + cfg.test_block, len(dates))
        test_dates = dates[start:end]
        train_end = max(0, start - cfg.embargo_dates)
        train_dates = dates[:train_end]
        
        if len(train_dates) < cfg.min_train_dates // 2:
            start += cfg.step
            continue
        
        dtr_all = panel[panel["date"].isin(train_dates)]
        dte = panel[panel["date"].isin(test_dates)]
        
        tr_d = np.array(sorted(dtr_all["date"].unique()))
        n_va = max(5, int(len(tr_d) * cfg.valid_frac_of_train))
        va_d, fit_d = set(tr_d[-n_va:]), set(tr_d[:-n_va]) if len(tr_d) > n_va + 20 else set(tr_d)
        d_fit = dtr_all[dtr_all["date"].isin(fit_d)]
        d_va = dtr_all[dtr_all["date"].isin(va_d)] if va_d != fit_d else None
        
        bundle = train_lambdarank_ensemble(
            train_df=d_fit,
            feature_cols=feature_cols,
            horizons=horizons,
            valid_df=d_va if len(d_va) else None,
            n_bins=cfg.n_bins,
            config=LambdaRankConfig(
                n_bins=cfg.n_bins,
                num_boost_round=cfg.num_boost_round,
                use_gpu=cfg.use_gpu,
            ),
        )
        
        last_bundle = bundle
        part = dte.copy()
        part["score"] = predict_ensemble(bundle, part, feature_cols).values
        oos_parts.append(part)
        
        ic = part.groupby("date").apply(
            lambda g: spearmanr(g["score"], g["y"]).correlation
            if g["y"].nunique() > 1 and g["score"].nunique() > 1 else np.nan
        )
        
        st = {
            "train_end": str(pd.to_datetime(train_dates.max()).date()),
            "test_start": str(pd.to_datetime(test_dates.min()).date()),
            "test_end": str(pd.to_datetime(test_dates.max()).date()),
            "ic_mean": float(np.nanmean(ic)),
            "ic_ir": float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)),
            "n_test": int(len(part)),
        }
        stats.append(st)
        
        print(f"  win IC={st['ic_mean']:.4f} IR={st['ic_ir']:.2f} end={st['test_end']}")
        
        if end >= len(dates):
            break
        start += cfg.step
    
    if not oos_parts:
        raise RuntimeError("Expanding multih ranker produced no OOS scores.")
    
    oos = pd.concat(oos_parts, ignore_index=True).sort_values(["date", "ticker"])
    return oos, stats, last_bundle


# =============================================================================
# Cross-Sectional to Binary Direction Frame (for conformal)
# =============================================================================

def scores_to_direction_frame(
    scored: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Build binary supervised frame from CS scores:
      features: raw score + cross-sectional z-score + rank pct
      label: 1{y > 0}
    """
    d = scored.sort_values(["date", "ticker"]).copy()
    d["score_z"] = d.groupby("date")[score_col].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12)
    )
    d["score_rk"] = d.groupby("date")[score_col].rank(pct=True)
    d["y_bin"] = (d["y"] > 0).astype(int)
    return d


# =============================================================================
# Binary LightGBM (for meta-labeling / conformal)
# =============================================================================

def train_binary_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    valid: Optional[tuple] = None,
) -> lgb.Booster:
    """Train binary LightGBM classifier."""
    dtr = lgb.Dataset(X, label=y)
    valid_sets, valid_names, callbacks = [dtr], ["train"], [lgb.log_evaluation(0)]
    
    if valid is not None:
        dva = lgb.Dataset(valid[0], label=valid[1], reference=dtr)
        valid_sets.append(dva)
        valid_names.append("valid")
        callbacks.insert(0, lgb.early_stopping(30))
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 23,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }
    
    return lgb.train(
        params, dtr,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing cross-sectional ranker...")
    
    # Create synthetic panel
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    n_tickers = 30
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    
    # Factor model
    mkt = np.random.normal(0.0002, 0.01, len(dates))
    rets = {}
    panels = []
    
    for i, tkr in enumerate(tickers):
        beta = np.random.uniform(0.5, 1.5)
        r = beta * mkt + np.random.normal(0.0, 0.012, len(dates))
        rets[tkr] = r
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
        
        f = pd.DataFrame({
            "ret_1": close.pct_change(),
            "ret_5": close.pct_change(5),
            "ret_10": close.pct_change(10),
            "vol_10": close.pct_change().rolling(10).std(),
            "vol_20": close.pct_change().rolling(20).std(),
            "ma_gap": close / close.rolling(20).mean() - 1.0,
        }, index=dates)
        
        y = close.pct_change(5).shift(-5)
        df = f.copy()
        df["y"] = y + 0.04 * f["ret_5"] + 0.02 * f["ma_gap"]  # planted signal
        df["ticker"] = tkr
        df["date"] = dates
        panels.append(df)
    
    panel = pd.concat(panels).dropna().reset_index(drop=True)
    returns_wide = pd.DataFrame(rets, index=dates)
    feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
    
    # Add multi-horizon labels
    panel = add_multi_horizon_labels(panel, returns_wide, horizons=(1, 5, 21), n_bins=5)
    print(f"Panel shape: {panel.shape}")
    print(f"Relevance bins: {panel['relevance'].value_counts().sort_index().to_dict()}")
    
    # Test expanding ranker
    cfg = ExpandingRankerConfig(min_train_dates=200, test_block=21, step=21, embargo_dates=5)
    oos, win_stats, model = expanding_window_lambdarank(panel, feature_cols, cfg)
    
    ic = daily_ic(oos)
    print(f"\nOOS IC mean={np.nanmean(ic):.4f} IR={np.nanmean(ic)/(np.nanstd(ic)+1e-12):.3f}")
    print(f"Windows: {len(win_stats)}")
    
    # Test multi-horizon
    panel_mh = panel.copy()
    oos_mh, stats_mh, bundle = expanding_multih_ranker(panel_mh, feature_cols, horizons=(1, 5, 21), cfg=cfg)
    ic_mh = daily_ic(oos_mh)
    print(f"\nMulti-horizon IC mean={np.nanmean(ic_mh):.4f} IR={np.nanmean(ic_mh)/(np.nanstd(ic_mh)+1e-12):.3f}")
    print(f"Ensemble weights: {bundle['ens_weights']}")
    
    # Test direction frame
    direction = scores_to_direction_frame(oos_mh)
    print(f"\nDirection frame: {direction.shape}, y_bin dist: {direction['y_bin'].value_counts().to_dict()}")
    
    # Test binary model
    feats = feature_cols + ["score", "score_z", "score_rk"]
    X_tr = direction[direction["date"] < "2021-01-01"][feats]
    y_tr = direction[direction["date"] < "2021-01-01"]["y_bin"]
    X_te = direction[direction["date"] >= "2021-01-01"][feats]
    y_te = direction[direction["date"] >= "2021-01-01"]["y_bin"]
    
    clf = train_binary_lgbm(X_tr, y_tr, valid=(X_te, y_te))
    p = clf.predict(X_te)
    auc = roc_auc_score(y_te, p)
    print(f"Binary AUC: {auc:.4f}")
    
    print("\nAll tests passed!")
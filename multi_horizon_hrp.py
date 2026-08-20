"""
Multi-Horizon Relevance and Sector-Neutral HRP (Enhanced)

This module extends the existing implementations with:
1. Multi-horizon LambdaRank ensemble with proper blending
2. Advanced sector-neutral HRP with factor neutralization
3. Borrow-aware position caps
4. Beta/position constraints
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# =============================================================================
# Multi-Horizon Relevance
# =============================================================================

@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon relevance."""
    horizons: Tuple[int, ...] = (1, 5, 21)
    n_bins: int = 5
    horizon_weights: Optional[Dict[int, float]] = None  # If None, use 1/sqrt(h)
    blend_weight: float = 0.4  # Weight for blended relevance model
    
    def __post_init__(self):
        if self.horizon_weights is None:
            raw = {h: 1.0 / np.sqrt(h) for h in self.horizons}
            s = sum(raw.values())
            self.horizon_weights = {h: raw[h] / s for h in self.horizons}


def cs_relevance_from_y(y: pd.Series, n_bins: int = 5) -> pd.Series:
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


def add_multi_horizon_labels(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    config: Optional[MultiHorizonConfig] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add multi-horizon forward returns and relevance bins.
    
    Args:
        panel: Long-format panel with date, ticker, features
        returns_wide: Wide returns (date x ticker) for computing forward returns
        config: MultiHorizonConfig
        price_col: If returns_wide not provided, use price column to compute
        
    Returns:
        Panel with added y_h{h}, rel_h{h}, y (primary), relevance (blended)
    """
    config = config or MultiHorizonConfig()
    d = panel.copy()
    horizons = config.horizons
    n_bins = config.n_bins
    
    # Compute forward returns if not present
    rw = returns_wide.sort_index()
    
    for h in horizons:
        col = f"y_h{h}"
        if col in d.columns:
            continue
        # Compound h-day return starting NEXT day: (t+1 ... t+h)
        fwd = (1.0 + rw).rolling(h).apply(np.prod, raw=True).shift(-h) - 1.0
        long = fwd.stack().rename(col)
        long.index = long.index.set_names(["date", "ticker"])
        tmp = long.reset_index()
        tmp["date"] = pd.to_datetime(tmp["date"])
        d = d.merge(tmp, on=["date", "ticker"], how="left")
    
    # Primary y = medium horizon (5d) or middle
    primary_h = 5 if 5 in horizons else horizons[len(horizons) // 2]
    d["y"] = d[f"y_h{primary_h}"]
    
    # Per-horizon relevance bins
    rel_cols = []
    for h in horizons:
        rc = f"rel_h{h}"
        d[rc] = d.groupby("date")[f"y_h{h}"].transform(
            lambda s, nb=n_bins: cs_relevance_from_y(s, nb)
        )
        rel_cols.append(rc)
    
    # Blended relevance (continuous then rounded)
    blend = np.zeros(len(d), dtype=float)
    for h in horizons:
        blend += config.horizon_weights[h] * d[f"rel_h{h}"].astype(float).values
    
    d["relevance"] = np.clip(np.rint(blend), 0, n_bins - 1).astype(int)
    d["relevance_blend_raw"] = blend
    d.attrs["horizon_weights"] = config.horizon_weights
    d.attrs["horizons"] = horizons
    
    return d.dropna(subset=["y", "relevance"] + [f"y_h{h}" for h in horizons])


# =============================================================================
# Multi-Horizon LambdaRank Ensemble
# =============================================================================

@dataclass
class LambdaRankConfig:
    """Configuration for LambdaRank training."""
    n_bins: int = 5
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 25
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 2.0
    early_stopping_rounds: int = 40
    eval_at: Tuple[int, ...] = (5, 10, 20)
    lambdarank_truncation_level: int = 30
    verbose: int = -1
    use_gpu: bool = False
    num_boost_round: int = 300
    num_threads: Optional[int] = None


def _groups_from_dates(dates: pd.Series) -> np.ndarray:
    """LightGBM group array from date series (must be sorted)."""
    return dates.groupby(dates, sort=False).size().to_numpy()


def _fit_lambdarank(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    rel_col: str,
    config: LambdaRankConfig,
) -> lgb.Booster:
    """Train single LambdaRank model."""
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
        # Use every core by default. An earlier default of cpu_count()//2 came from a
        # toy benchmark (200 groups x 100 rows) where 6 threads beat 12; at the real
        # ranker shape the ordering INVERTS. Measured, 735,000 rows / 1,500 groups /
        # 200 rounds on 12 logical cores:
        #     4 threads 28.41s | 6 threads 22.63s | 8 threads 21.29s
        #    10 threads 22.18s | 12 threads 18.49s  <- fastest
        # Small-data thread benchmarks do not transfer: sync overhead dominates when
        # there is little work per thread, and that is not this workload.
        "num_threads": config.num_threads or (os.cpu_count() or 4),
        "force_row_wise": True,  # skips LightGBM's own row/col-wise probe overhead
    }
    if config.use_gpu:
        # NOTE: this requires a LightGBM built with -DUSE_GPU=1. The wheel installed
        # here is NOT: it raises "GPU Tree Learner was not enabled in this build".
        # Verify before enabling, otherwise training dies at the first train() call.
        params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    
    return lgb.train(
        params, dtr,
        num_boost_round=config.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def train_multih_lambdarank_ensemble(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    config: MultiHorizonConfig,
    valid_df: Optional[pd.DataFrame] = None,
    ranker_config: Optional[LambdaRankConfig] = None,
) -> Dict:
    """
    Train LambdaRank ensemble: one per horizon + blended.
    
    Returns:
        Dict with 'models', 'ens_weights', 'horizons', 'horizon_weights'
    """
    ranker_config = ranker_config or LambdaRankConfig(n_bins=config.n_bins)
    
    models = {}
    
    # Train per-horizon models
    for h in config.horizons:
        models[f"h{h}"] = _fit_lambdarank(
            train_df, valid_df, feature_cols, f"rel_h{h}", ranker_config
        )
    
    # Train blended model
    models["blend"] = _fit_lambdarank(
        train_df, valid_df, feature_cols, "relevance", ranker_config
    )
    
    # Ensemble weights: horizon weights * (1 - blend_weight) + blend_weight for blend
    hw = config.horizon_weights
    ens = {f"h{h}": (1 - config.blend_weight) * hw[h] for h in config.horizons}
    ens["blend"] = config.blend_weight
    z = sum(ens.values())
    ens = {k: v / z for k, v in ens.items()}
    
    return {
        "models": models,
        "ens_weights": ens,
        "horizons": config.horizons,
        "horizon_weights": config.horizon_weights,
    }


def predict_multih_ensemble(
    bundle: Dict,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """Predict using multi-horizon ensemble."""
    acc = np.zeros(len(df), dtype=float)
    for tag, w in bundle["ens_weights"].items():
        m = bundle["models"][tag]
        acc += w * m.predict(df[feature_cols], num_iteration=m.best_iteration)
    return pd.Series(acc, index=df.index, name="score")


# =============================================================================
# Expanding Window Multi-Horizon Ranker
# =============================================================================

@dataclass
class ExpandingRankerConfig:
    """Configuration for expanding window ranker."""
    min_train_dates: int = 252
    test_block: int = 21
    step: int = 21
    embargo_dates: int = 5
    valid_frac_of_train: float = 0.15
    n_bins: int = 5
    num_boost_round: int = 300
    use_gpu: bool = False


def expanding_multih_ranker(
    panel: pd.DataFrame,
    feature_cols: List[str],
    mh_config: Optional[MultiHorizonConfig] = None,
    exp_config: Optional[ExpandingRankerConfig] = None,
    ranker_config: Optional[LambdaRankConfig] = None,
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Expanding window multi-horizon LambdaRank.
    
    Returns:
        (OOS scores DataFrame, window stats list, last ensemble bundle)
    """
    mh_config = mh_config or MultiHorizonConfig()
    exp_config = exp_config or ExpandingRankerConfig()
    ranker_config = ranker_config or LambdaRankConfig(n_bins=mh_config.n_bins)
    
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    dates = np.array(sorted(panel["date"].unique()))
    
    oos_parts = []
    window_stats = []
    last_bundle = None
    
    start = exp_config.min_train_dates
    while start < len(dates):
        end = min(start + exp_config.test_block, len(dates))
        test_dates = dates[start:end]
        train_end = max(0, start - exp_config.embargo_dates)
        train_dates = dates[:train_end]
        
        if len(train_dates) < exp_config.min_train_dates // 2:
            start += exp_config.step
            continue
        
        dtr_all = panel[panel["date"].isin(train_dates)]
        dte = panel[panel["date"].isin(test_dates)]
        
        tr_d = np.array(sorted(dtr_all["date"].unique()))
        n_va = max(5, int(len(tr_d) * exp_config.valid_frac_of_train))
        va_d, fit_d = set(tr_d[-n_va:]), set(tr_d[:-n_va]) if len(tr_d) > n_va + 20 else set(tr_d)
        d_fit = dtr_all[dtr_all["date"].isin(fit_d)]
        d_va = dtr_all[dtr_all["date"].isin(va_d)] if va_d != fit_d else None
        
        bundle = train_multih_lambdarank_ensemble(
            d_fit, feature_cols, mh_config, valid_df=d_va, ranker_config=ranker_config
        )
        last_bundle = bundle
        
        part = dte.copy()
        part["score"] = predict_multih_ensemble(bundle, part, feature_cols).values
        oos_parts.append(part)
        
        ic = part.groupby("date").apply(
            lambda g: g["score"].corr(g["y"])
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
        window_stats.append(st)
        print(f"  win train_end={st['train_end']} IC={st['ic_mean']:.4f} IR={st['ic_ir']:.2f}")
        
        if end >= len(dates):
            break
        start += exp_config.step
    
    if not oos_parts:
        raise RuntimeError("Expanding ranker produced no OOS scores")
    
    oos = pd.concat(oos_parts, ignore_index=True).sort_values(["date", "ticker"])
    return oos, window_stats, last_bundle


# =============================================================================
# Sector-Neutral HRP with Factor Neutralization
# =============================================================================

@dataclass
class SectorNeutralConfig:
    """Configuration for sector-neutral HRP."""
    neutralize_sizes: bool = True
    conf_blend: float = 0.3
    gross_target: float = 1.0
    max_name_weight: float = 0.05
    max_short_weight: float = 0.03
    max_participation: float = 0.05
    borrow_soft_bps: float = 150.0
    borrow_hard_bps: float = 500.0
    book_nav: float = 1e7


def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Correlation distance matrix."""
    return np.sqrt((1.0 - corr.clip(-1, 1)) / 2.0)


def _quasi_diag(link: np.ndarray) -> List:
    """Quasi-diagonalization of a linkage matrix, returning leaf order.

    Iterative stack walk over the linkage tree instead of the classic
    pandas-Series version, which rebuilt and re-sorted an index on every pass and
    showed up as the largest remaining cost in this function after _cluster_var was
    made positional. Same output: leaves left-to-right in cluster order.
    """
    link = link.astype(int)
    n_items = int(link[-1, 3])
    n_leaves = link.shape[0] + 1
    order: List[int] = []
    stack = [2 * link.shape[0]]  # root node id in scipy's numbering
    while stack:
        node = stack.pop()
        if node < n_leaves:
            order.append(node)
            continue
        row = link[node - n_leaves]
        # push right then left so the left subtree is emitted first
        stack.append(int(row[1]))
        stack.append(int(row[0]))
    return order[:n_items]


def _cluster_var(cov: np.ndarray, items: np.ndarray) -> float:
    """Inverse-variance cluster variance for HRP, positional.

    Takes a raw numpy covariance matrix and integer positions, NOT a labelled
    DataFrame. The label form (cov.loc[items, items]) made this the pipeline's
    bottleneck: it is the innermost step of the recursive bisection below, so it
    runs for every cluster split, on every date, for every sector sleeve, and each
    call paid full pandas index alignment plus dtype inference. A py-spy dump of a
    stalled run showed the entire stack under this line was pandas machinery
    (__getitem__ -> _getitem_lowerdim -> get_indexer -> is_numeric_dtype ->
    pandas_dtype) rather than arithmetic. np.ix_ does the same slice positionally.
    """
    sub = cov[np.ix_(items, items)]
    w = 1.0 / np.diag(sub)
    w /= w.sum()
    return float(w @ sub @ w)


_CUDA_MIN_NAMES = 1000
"""Universe size at which GPU covariance starts winning.

Measured on this box (MX550, 120-day window, float32), corr of an n-column frame:

      n   pandas    numpy    cuda
    497    40.5ms   39.1ms   79.7ms   <- CPU wins, transfer dominates
   1000   548.5ms   20.4ms   14.1ms
   2500  3914.0ms  165.4ms   17.3ms
   5000 16006.6ms  839.2ms  206.5ms
  10000 63493.8ms 3402.1ms  364.6ms   <- 174x vs pandas, 9x vs numpy

Below ~1000 names the host<->device copy costs more than the matmul saves, so
routing everything to the GPU would be a REGRESSION at the current 497-name
universe. Above it the GPU wins by an order of magnitude, which is what makes the
10k universe tractable at all.
"""


def _cov_corr(rets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Covariance and correlation of a wide return frame, as raw numpy.

    pandas' DataFrame.cov()/.corr() are pathologically slow on wide frames: they
    walk column pairs, so cost explodes with n (63.5s at n=10000 vs 3.4s for the
    same numpy matmul). Both matrices come from ONE centred matmul here, and the
    correlation is derived from the covariance instead of recomputing it.

    Returns numpy, not DataFrames, because every consumer downstream indexes
    positionally; handing back a labelled frame would reintroduce the pandas
    indexing cost the callers were just freed from. Column order is preserved,
    so caller-side `rets.columns` remains the label mapping.

    float64 throughout on CPU. The GPU path computes in float32 (the MX550 has no
    usable f64 compute for this, and DirectML lacks f64 sqrt entirely) and casts
    back, which is fine for a correlation used only to build a clustering tree.
    """
    X = rets.to_numpy(dtype=np.float64, copy=False)
    T, n = X.shape
    if T < 2:
        z = np.zeros((n, n), dtype=np.float64)
        return z, z

    if n >= _CUDA_MIN_NAMES:
        try:
            import torch

            if torch.cuda.is_available():
                with torch.no_grad():
                    x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).cuda()
                    xc = x - x.mean(0, keepdim=True)
                    cov_t = (xc.T @ xc) / (T - 1)
                    sd = torch.sqrt(torch.diag(cov_t)).clamp_min(1e-12)
                    corr_t = cov_t / torch.outer(sd, sd)
                    cov = cov_t.double().cpu().numpy()
                    corr = corr_t.double().cpu().numpy()
                del x, xc, cov_t, corr_t, sd
                torch.cuda.empty_cache()
                np.clip(corr, -1.0, 1.0, out=corr)
                return cov, corr
        except Exception:
            pass  # no CUDA / OOM / driver issue: fall through to numpy

    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (T - 1)
    sd = np.sqrt(np.diag(cov))
    np.maximum(sd, 1e-12, out=sd)
    corr = cov / np.outer(sd, sd)
    np.clip(corr, -1.0, 1.0, out=corr)
    return cov, corr


def hrp_weights_from_returns(returns: pd.DataFrame) -> pd.Series:
    """HRP weights from return matrix."""
    rets = returns.dropna(axis=1, how="any")
    if rets.shape[1] <= 1:
        return pd.Series(1.0, index=rets.columns) if rets.shape[1] == 1 else pd.Series(dtype=float)

    cols = rets.columns
    cov_v, corr_v = _cov_corr(rets)
    dist = np.sqrt(np.maximum((1.0 - corr_v) / 2.0, 0.0))
    np.fill_diagonal(dist, 0.0)
    link = sch.linkage(ssd.squareform(dist, checks=False), method="single")

    # Bisect over integer POSITIONS into cov_v, so the hot loop never touches a
    # pandas index. _quasi_diag already returns positions into the same ordering.
    order_ix = np.asarray(_quasi_diag(link), dtype=np.intp)
    w_v = np.ones(len(cov_v), dtype=float)

    clusters = [order_ix]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            c1, c2 = cl[:half], cl[half:]
            v1, v2 = _cluster_var(cov_v, c1), _cluster_var(cov_v, c2)
            a = 1.0 - v1 / (v1 + v2 + 1e-12)
            w_v[c1] *= a
            w_v[c2] *= (1.0 - a)
            nxt += [c1, c2]
        clusters = nxt

    w = pd.Series(w_v[order_ix], index=cols[order_ix])

    return (w / w.sum()).reindex(returns.columns).fillna(0.0)


def estimate_betas(
    returns_wide: pd.DataFrame,
    lookback: int = 60,
    market_col: Optional[str] = None,
) -> pd.Series:
    """Rolling beta vs market (or equal-weight portfolio), vectorised.

    One closed-form OLS slope for every name at once: beta_i = cov(y_i, m)/var(m).
    The previous version looped over columns, pulling each one out of the DataFrame
    (returns_wide[col] -> _get_item -> _box_col_values -> arrow __getitem__) and
    fitting a separate sklearn LinearRegression per name. py-spy caught the book
    backtest sitting in exactly that loop: with ~500 names re-fit on every one of
    ~1,900 dates it was the dominant cost of the whole backtest.

    NaN handling matches the loop it replaces: a name needs >= 20 overlapping
    finite observations with the market, otherwise its beta is 1.0. Names with zero
    market variance also fall back to 1.0.
    """
    rw = returns_wide.dropna(how="all").iloc[-lookback:]
    if rw.shape[1] == 0:
        return pd.Series(dtype=float)
    mkt = rw[market_col] if market_col and market_col in rw.columns else rw.mean(axis=1)

    Y = rw.to_numpy(dtype=np.float64, copy=False)
    m = np.asarray(mkt, dtype=np.float64)

    valid = np.isfinite(Y) & np.isfinite(m)[:, None]
    n_obs = valid.sum(axis=0)

    # Per-column means over the valid overlap only (zeros where invalid).
    Yz = np.where(valid, Y, 0.0)
    Mz = np.where(valid, m[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = np.maximum(n_obs, 1)
        y_bar = Yz.sum(axis=0) / cnt
        m_bar = Mz.sum(axis=0) / cnt
        yc = np.where(valid, Y - y_bar, 0.0)
        mc = np.where(valid, m[:, None] - m_bar, 0.0)
        cov = (yc * mc).sum(axis=0)
        var = (mc * mc).sum(axis=0)
        beta = np.divide(cov, var, out=np.ones_like(cov), where=var > 0)

    beta = np.where(n_obs >= 20, beta, 1.0)
    beta = np.where(np.isfinite(beta), beta, 1.0)
    return pd.Series(beta, index=rw.columns)


def sector_neutralize_scores(
    scores: pd.Series,
    sectors: pd.Series,
    method: str = "demean",
) -> pd.Series:
    """
    Cross-sectional sector neutralization of scores.
    
    Args:
        scores: Scores indexed by ticker
        sectors: Sector assignments indexed by ticker
        method: 'demean' or 'zscore'
        
    Returns:
        Sector-neutralized scores
    """
    df = pd.DataFrame({"s": scores, "sec": sectors}).dropna(subset=["s"])
    if method == "zscore":
        out = df.groupby("sec")["s"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )
    else:
        out = df.groupby("sec")["s"].transform(lambda x: x - x.mean())
    return out.reindex(scores.index)


def project_weights_factor_neutral(
    w: pd.Series,
    betas: pd.Series,
    sectors: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Project weights to dollar-neutral and beta-neutral (+ sector-neutral).
    
    Iteratively projects out: mean (dollar neutral), beta, sector means.
    """
    idx = w.index
    w = w.fillna(0.0).astype(float)
    b = betas.reindex(idx).fillna(1.0)
    
    for _ in range(3):
        # Dollar neutral
        w = w - w.mean()
        
        # Beta neutral
        bb = b.values
        wv = w.values
        num = wv @ bb
        den = bb @ bb + 1e-12
        w = pd.Series(wv - (num / den) * bb, index=idx)
        
        if sectors is not None:
            sec = sectors.reindex(idx).astype(str).fillna("UNK")
            tmp = pd.DataFrame({"w": w, "sec": sec})
            w = tmp.groupby("sec")["w"].transform(lambda x: x - x.mean())
    
    return w


def apply_borrow_adv_caps(
    w: pd.Series,
    adv: pd.Series,
    borrow_bps_annual: pd.Series,
    config: SectorNeutralConfig,
) -> pd.Series:
    """
    Apply borrow and ADV participation caps.
    
    - Hard-to-borrow (above borrow_hard_bps): no shorts
    - Soft-to-borrow (above borrow_soft_bps): shrink shorts
    - Per-name weight caps
    - ADV participation caps
    """
    w = w.copy().astype(float)
    adv = adv.reindex(w.index).fillna(config.book_nav)
    br = borrow_bps_annual.reindex(w.index).fillna(100.0)
    
    # Hard HTB: zero out shorts
    w[(w < 0) & (br >= config.borrow_hard_bps)] = 0.0
    
    # Soft HTB: shrink shorts
    soft = (w < 0) & (br > config.borrow_soft_bps) & (br < config.borrow_hard_bps)
    if soft.any():
        scale = 1.0 - (br[soft] - config.borrow_soft_bps) / (
            config.borrow_hard_bps - config.borrow_soft_bps + 1e-12
        )
        w.loc[soft] = w.loc[soft] * scale.clip(0.1, 1.0)
    
    # Per-name caps
    w = w.clip(lower=-config.max_short_weight, upper=config.max_name_weight)
    
    # ADV participation cap
    max_w_adv = (config.max_participation * adv / config.book_nav).clip(lower=1e-4)
    w = w.clip(lower=-max_w_adv, upper=max_w_adv)
    
    return w


def build_sector_neutral_hrp_weights(
    day_sizes: pd.Series,
    sectors: pd.Series,
    returns_hist: pd.DataFrame,
    config: Optional[SectorNeutralConfig] = None,
    betas: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Build sector-neutral HRP weights from conformal sizes.
    
    Process:
    1. Optional sector-demean conformal sizes
    2. Split into L/S sleeves
    3. HRP within each sleeve
    4. Blend HRP structure with conviction
    5. Project to factor/sector/dollar neutral
    6. Apply borrow/ADV caps
    6. Re-project and scale to gross target
    """
    config = config or SectorNeutralConfig()
    s = day_sizes.dropna().astype(float)
    sec = sectors.reindex(s.index)
    
    if config.neutralize_sizes:
        s = sector_neutralize_scores(s, sec, method="demean").fillna(0.0)
    
    # Estimate betas if not provided
    if betas is None:
        betas = estimate_betas(returns_hist)
    
    w = pd.Series(0.0, index=s.index)
    rh = returns_hist[[c for c in returns_hist.columns if c in s.index]]
    
    def sleeve(names: pd.Series, side: str) -> pd.Series:
        if names.empty:
            return pd.Series(dtype=float)
        cols = [c for c in names.index if c in rh.columns]
        if len(cols) <= 1:
            ew = pd.Series(1.0 / len(names), index=names.index)
        else:
            ew = hrp_weights_from_returns(rh[cols].dropna(how="any"))
            ew = ew.reindex(names.index).fillna(0.0)
            ew = ew / ew.sum() if ew.sum() > 0 else pd.Series(1.0 / len(names), index=names.index)
        
        mag = names.abs()
        mag = mag / (mag.sum() + 1e-12)
        blend = (1 - config.conf_blend) * ew + config.conf_blend * mag.reindex(ew.index).fillna(0.0)
        blend = blend / (blend.sum() + 1e-12)
        out = blend * float(names.abs().sum())
        return out if side == "long" else -out
    
    longs, shorts = s[s > 0], s[s < 0]

    # The two sleeves ARE independent (separate correlation matrices, no shared state
    # until the assignments below), so this looks like the one parallelisable spot
    # inside a date. Measured, it is not: a ThreadPoolExecutor(2) over the two sleeves
    # ran 0.76x -- SLOWER on every date tried (33->40ms, 31->40ms, 52->75ms).
    # After the numpy/positional work each sleeve is only ~15ms, and thread startup
    # plus the contended BLAS pool cost more than the overlap saves. Parallelism at
    # this granularity is below the useful threshold; the win is at the date level
    # (v5_integrated's process pool, 4.15x). Left serial deliberately.
    wl, ws = sleeve(longs, "long"), sleeve(shorts, "short")
    
    if len(wl):
        w.loc[wl.index] = wl.values
    if len(ws):
        w.loc[ws.index] = ws.values
    
    # Factor/sector/dollar neutralization
    w = project_weights_factor_neutral(w, betas, sectors=sec)
    
    if w.abs().sum() > 0:
        w = w * (config.gross_target / w.abs().sum())
    
    # Apply borrow/ADV caps (need adv and borrow passed via config)
    # This is a placeholder - in practice pass adv/borrow
    return w.fillna(0.0)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing multi-horizon relevance...")
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=400)
    tickers = [f"T{i:02d}" for i in range(10)]
    
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
        
        df = f.copy()
        df["ticker"] = tkr
        df["date"] = dates
        panels.append(df.reset_index(drop=True))
    
    panel = pd.concat(panels, ignore_index=True)
    returns_wide = pd.DataFrame(rets, index=dates)
    
    # Add multi-horizon labels
    mh_config = MultiHorizonConfig(horizons=(1, 5, 21), n_bins=5)
    panel = add_multi_horizon_labels(panel, returns_wide, mh_config)
    
    print(f"Panel shape: {panel.shape}")
    print(f"Horizons: {panel.attrs['horizons']}")
    print(f"Relevance distribution:\n{panel['relevance'].value_counts().sort_index()}")
    print(f"Per-horizon relevance:")
    for h in mh_config.horizons:
        print(f"  rel_h{h}: {panel[f'rel_h{h}'].value_counts().sort_index().to_dict()}")
    
    # Test multi-horizon ensemble training
    print("\nTesting multi-horizon ensemble...")
    feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
    
    # Split for quick test
    train_dates = dates[:200]
    test_dates = dates[200:220]
    dtr = panel[panel["date"].isin(train_dates)]
    dte = panel[panel["date"].isin(test_dates)]
    
    bundle = train_multih_lambdarank_ensemble(
        dtr, feature_cols, mh_config,
        ranker_config=LambdaRankConfig(num_boost_round=50, n_bins=5)
    )
    
    print(f"Ensemble weights: {bundle['ens_weights']}")
    
    # Predict
    scores = predict_multih_ensemble(bundle, dte, feature_cols)
    dte = dte.copy()
    dte["score"] = scores.values
    
    ic = dte.groupby("date").apply(
        lambda g: g["score"].corr(g["y"])
        if g["y"].nunique() > 1 and g["score"].nunique() > 1 else np.nan
    )
    print(f"Test IC mean: {np.nanmean(ic):.4f}")
    
    # Test sector-neutral HRP
    print("\nTesting sector-neutral HRP...")
    sectors = pd.Series({t: f"S{i % 3}" for i, t in enumerate(tickers)})
    day_sizes = dte[dte["date"] == dte["date"].iloc[0]].set_index("ticker")["score"]
    # Add some variation for shorts
    day_sizes = day_sizes - day_sizes.median()
    
    sn_config = SectorNeutralConfig(
        neutralize_sizes=True,
        conf_blend=0.3,
        gross_target=1.0,
    )
    
    # Use trailing returns
    hist = returns_wide.iloc[150:200]
    betas = estimate_betas(hist)
    
    w = build_sector_neutral_hrp_weights(
        day_sizes, sectors, hist, config=sn_config, betas=betas
    )
    
    print(f"Weights: {w[w != 0].to_dict()}")
    print(f"Gross: {w.abs().sum():.4f}, Net: {w.sum():.4f}")
    print(f"Sector exposure:")
    sec_exp = pd.DataFrame({"w": w, "sec": sectors.reindex(w.index)}).groupby("sec")["w"].sum()
    print(f"  {sec_exp.to_dict()}")
    print(f"Beta exposure: {(w * betas.reindex(w.index).fillna(1)).sum():.4f}")
    
    print("\nAll tests passed!")
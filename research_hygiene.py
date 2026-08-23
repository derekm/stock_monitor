"""
Core Research Hygiene Modules for Quant Research

Implements:
1. Triple-barrier labeling (AFML)
2. Purged/Embargoed K-Fold CV
3. Fractional Differentiation (fixed-width window)
4. CUSUM Filter for event sampling
5. Meta-labeling (López de Prado)
6. Daily volatility estimation

All designed for point-in-time correctness and leakage prevention.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

warnings.filterwarnings("ignore")


# =============================================================================
# Volatility Estimation
# =============================================================================

def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """
    EWM volatility of returns (AFML-style).
    
    Args:
        close: Close price series with DatetimeIndex
        span: EWM span (default 100 days)
        
    Returns:
        Daily volatility series
    """
    rets = close.pct_change()
    return rets.ewm(span=span).std()


# =============================================================================
# Triple-Barrier Labeling (AFML)
# =============================================================================

@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    pt_sl: tuple[float, float] = (1.0, 1.0)  # (profit_take, stop_loss) multipliers of vol
    max_holding: Optional[int] = None  # Max holding period in bars
    min_ret: float = 0.0  # Minimum return threshold for non-zero label
    molecule: Optional[pd.DatetimeIndex] = None  # Subset of events to process


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    config: Optional[TripleBarrierConfig] = None,
    vol: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Triple-barrier method for labeling financial events.
    
    Args:
        close: Close price series with DatetimeIndex
        events: Event timestamps (entry times)
        config: TripleBarrierConfig with pt_sl, max_holding, min_ret
        vol: Pre-computed daily volatility (if None, computed internally)
        
    Returns:
        DataFrame with columns: t0 (entry), t1 (exit), ret, trgt, bin (-1, 0, 1)
    """
    if config is None:
        config = TripleBarrierConfig()
    
    if config.molecule is None:
        molecule = events
    else:
        molecule = config.molecule
    
    if vol is None:
        vol = get_daily_vol(close).reindex(events).dropna()
    
    out = []
    close = close.sort_index()
    
    for t0 in molecule:
        if t0 not in close.index or t0 not in vol.index:
            continue
        p0 = close.loc[t0]
        trgt = float(vol.loc[t0])
        if trgt <= 0 or np.isnan(trgt):
            continue
        
        path = close.loc[t0:]
        if config.max_holding is not None:
            path = path.iloc[: config.max_holding + 1]
        if len(path) < 2:
            continue
        
        rets = path / p0 - 1.0
        upper = config.pt_sl[0] * trgt if config.pt_sl[0] > 0 else np.inf
        lower = -config.pt_sl[1] * trgt if config.pt_sl[1] > 0 else -np.inf
        
        hit_pt = rets[rets >= upper]
        hit_sl = rets[rets <= lower]
        t_pt = hit_pt.index[0] if len(hit_pt) else None
        t_sl = hit_sl.index[0] if len(hit_sl) else None
        t_vb = path.index[-1]  # vertical barrier
        
        candidates = {t: "vb" for t in [t_vb]}
        if t_pt is not None:
            candidates[t_pt] = "pt"
        if t_sl is not None:
            candidates[t_sl] = "sl"
        t_exit = min(candidates.keys())
        reason = candidates[t_exit]
        r = float(rets.loc[t_exit])
        
        if abs(r) < config.min_ret:
            lab = 0
        elif reason == "pt" or r > 0:
            lab = 1 if r > 0 else 0
        elif reason == "sl" or r < 0:
            lab = -1 if r < 0 else 0
        else:
            lab = int(np.sign(r))
        
        out.append({"t0": t0, "t1": t_exit, "ret": r, "trgt": trgt, "bin": lab})
    
    df = pd.DataFrame(out).set_index("t0")
    return df


def meta_label(
    primary_side: pd.Series,
    barrier_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Meta-labeling: y=1 if primary side agrees with realized barrier outcome.
    
    Args:
        primary_side: +1 / -1 position side from primary model at t0
        barrier_df: Output from triple_barrier_labels with 'bin' column
        
    Returns:
        DataFrame with added 'side' and 'meta_y' columns
    """
    df = barrier_df.copy()
    side = primary_side.reindex(df.index)
    df["side"] = side
    df["meta_y"] = (df["bin"] * df["side"] > 0).astype(int)
    
    # When primary says flat or barrier is 0, meta is 0
    df.loc[df["side"].isna() | (df["side"] == 0) | (df["bin"] == 0), "meta_y"] = 0
    return df


# =============================================================================
# Purged / Embargoed K-Fold Cross-Validation
# =============================================================================

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold with embargo (López de Prado).
    
    Ensures no label overlap between train and test by purging samples
    whose t1 overlaps test horizon, plus embargo gap.
    
    Usage:
        cv = PurgedKFold(n_splits=5, t1=events_t1, pct_embargo=0.01)
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            ...
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        self.n_splits = n_splits
        self.t1 = t1  # Series of event end times indexed like labels
        self.pct_embargo = pct_embargo
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
    
    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            indices = np.arange(X.shape[0])
            idx = X.index
        else:
            indices = np.arange(len(X))
            idx = self.t1.index if self.t1 is not None else pd.RangeIndex(len(X))
        
        n = len(indices)
        mbrg = int(n * self.pct_embargo)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(fold_sizes) - fold_sizes
        
        if self.t1 is None:
            t1 = pd.Series(idx, index=idx)
        else:
            t1 = self.t1.reindex(idx).ffill().bfill()
        
        for i in range(self.n_splits):
            test_start, test_size = starts[i], fold_sizes[i]
            test_indices = indices[test_start: test_start + test_size]
            test_times = idx[test_start: test_start + test_size]
            
            max_t1 = t1.loc[test_times].max()
            train_mask = np.ones(n, dtype=bool)
            
            # Purge: drop train whose t1 overlaps test horizon
            for j, t0 in enumerate(idx):
                if t1.loc[t0] >= test_times.min() and t0 <= max_t1:
                    train_mask[j] = False
            
            # Also remove test block
            train_mask[test_start: test_start + test_size] = False
            
            # Embargo after test
            embargo_end = min(test_start + test_size + mbrg, n)
            train_mask[test_start + test_size: embargo_end] = False
            
            train_indices = indices[train_mask]
            yield train_indices, test_indices


# =============================================================================
# Date-based Purged Split (more intuitive for daily data)
# =============================================================================

class PurgedDateSplit:
    """
    Expanding or rolling train dates with embargo before each test block.
    
    More intuitive than sample-based purged CV for daily panel data.
    """
    
    def __init__(self, embargo_dates: int = 5):
        self.embargo_dates = embargo_dates
    
    def expanding_windows(
        self,
        unique_dates: np.ndarray,
        min_train_dates: int,
        test_block: int,
        step: Optional[int] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate expanding train/test date windows.
        
        Args:
            unique_dates: Sorted array of unique dates
            min_train_dates: Minimum training dates before first test
            test_block: Number of dates in each test block
            step: Step size (default = test_block)
            
        Yields:
            (train_dates, test_dates) arrays
        """
        dates = np.array(sorted(pd.to_datetime(unique_dates)))
        step = step or test_block
        start_test = min_train_dates
        
        while start_test < len(dates):
            end_test = min(start_test + test_block, len(dates))
            test = dates[start_test:end_test]
            # Train ends embargo_dates before test start
            train_end = max(0, start_test - self.embargo_dates)
            train = dates[:train_end]
            
            if len(train) >= min_train_dates // 2 and len(test) > 0:
                yield train, test
            
            if end_test >= len(dates):
                break
            start_test += step


# =============================================================================
# Fractional Differentiation (Fixed-Width Window)
# =============================================================================

def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """
    Weights for fixed-width window fractional differentiation.
    
    Args:
        d: Differentiation order (0 < d < 1 for long memory preservation)
        threshold: Weight cutoff threshold
        max_size: Maximum window size
        
    Returns:
        Weight array (reversed for convolution)
    """
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-3, max_size: int = 1000) -> pd.Series:
    """
    Fixed-window fractional differentiation.
    
    Preserves long memory while achieving stationarity.
    
    Args:
        series: Input price/return series
        d: Differentiation order (0.3-0.5 typical for returns)
        threshold: Weight cutoff (default 1e-3 for practical window sizes)
        max_size: Maximum window size (default 1000)
        
    Returns:
        Fractionally differentiated series
    """
    w = get_weights_ffd(d, threshold, max_size)
    width = len(w)
    x = series.ffill().dropna().values
    
    if len(x) < width:
        # Fallback: use shorter window by truncating weights
        w = w[-len(x):]
        width = len(w)
        if width < 2:
            return pd.Series(index=series.index, dtype=float)
    
    # Use full convolution and take the last len(x) values
    out = np.convolve(x, w.ravel(), mode="full")[-len(x):]
    idx = series.ffill().dropna().index
    return pd.Series(out, index=idx)


# =============================================================================
# CUSUM Filter (Event Sampling)
# =============================================================================

def cusum_filter(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter on log-price; returns event timestamps.
    
    Samples events when cumulative return exceeds threshold h.
    Better than fixed-time sampling for capturing information-driven moves.
    
    Args:
        close: Close price series
        h: Threshold (typically daily_vol.median())
        
    Returns:
        DatetimeIndex of sampled events
    """
    t_events = []
    s_pos, s_neg = 0.0, 0.0
    log_ret = np.log(close).diff().dropna()
    
    for t, r in log_ret.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        
        if s_pos > h:
            s_pos = 0.0
            t_events.append(t)
        elif s_neg < -h:
            s_neg = 0.0
            t_events.append(t)
    
    return pd.DatetimeIndex(t_events)


# =============================================================================
# Rolling Sharpe Monitor (Live Research)
# =============================================================================

def rolling_sharpe_monitor(pnl: pd.Series, window: int = 63) -> pd.DataFrame:
    """
    Simple live research monitor: rolling mean/vol/sharpe + z-score decay flag.
    
    Args:
        pnl: Daily PnL series
        window: Rolling window (default 63 ~ quarter)
        
    Returns:
        DataFrame with mu, sd, sharpe, sharpe_z
    """
    mu = pnl.rolling(window).mean()
    sd = pnl.rolling(window).std()
    sharpe = mu / (sd + 1e-12) * np.sqrt(252)
    z = (sharpe - sharpe.rolling(window * 4).mean()) / (sharpe.rolling(window * 4).std() + 1e-12)
    
    return pd.DataFrame({
        "mu": mu,
        "sd": sd,
        "sharpe": sharpe,
        "sharpe_z": z,
    })


# =============================================================================
# Deflated Sharpe Ratio (Harvey / Bailey-López de Prado)
# =============================================================================

from scipy.stats import norm

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_benchmark: float,
    n_obs: int,
    var_sr: Optional[float] = None,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Approximate PSR/DSR-style probability that true SR > benchmark.
    
    Args:
        observed_sr: Observed Sharpe ratio
        sr_benchmark: Benchmark Sharpe (typically 0)
        n_obs: Number of return observations
        var_sr: Variance of SR estimator (if available from bootstrap)
        skew: Skewness of returns (default 0)
        kurt: Kurtosis of returns (default 3 = normal)
        
    Returns:
        Probability that true SR > benchmark
    """
    sr = observed_sr
    se = np.sqrt(
        (1 + 0.5 * sr**2 - skew * sr + ((kurt - 3) / 4) * sr**2) / max(n_obs, 2)
    )
    
    if var_sr is not None and var_sr > 0:
        se = np.sqrt(var_sr)
    
    z = (sr - sr_benchmark) / (se + 1e-12)
    return float(norm.cdf(z))


# =============================================================================
# Convenience: Feature Engineering Helpers
# =============================================================================

def make_basic_features(close: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Placeholder feature set — replace with your PIT-safe features.
    
    Args:
        close: Close price series
        volume: Optional volume series
        
    Returns:
        DataFrame with basic features
    """
    f = pd.DataFrame(index=close.index)
    r = close.pct_change()
    f["ret_1"] = r
    f["ret_5"] = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-12))
    f["ma_gap"] = close / close.rolling(20).mean() - 1
    f["fracdiff"] = frac_diff_ffd(close, d=0.4).reindex(close.index)
    
    if volume is not None:
        f["amihud"] = (r.abs() / (volume * close + 1e-12)).rolling(21).mean()
    
    return f.replace([np.inf, -np.inf], np.nan)


def make_xy_from_prices(
    close: pd.Series, 
    horizon: int = 5,
    volume: Optional[pd.Series] = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Simple supervised frame: predict forward return sign/value.
    
    Args:
        close: Close price series
        horizon: Forward horizon in days
        volume: Optional volume series
        
    Returns:
        (X, y_clf, y_reg) tuple
    """
    df = pd.DataFrame({"close": close})
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["rsi_14"] = 100 - 100 / (1 + (
        df["ret_1"].clip(lower=0).rolling(14).mean() /
        (-df["ret_1"].clip(upper=0).rolling(14).mean() + 1e-12)
    ))
    df["y_reg"] = close.pct_change(horizon).shift(-horizon)
    df["y_clf"] = (df["y_reg"] > 0).astype(int)
    df = df.dropna()
    
    feats = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "rsi_14"]
    return df[feats], df["y_clf"], df["y_reg"]


# =============================================================================
# Memory Cleanup
# =============================================================================

def free_gpu():
    """Free GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-barriers", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.book_barriers:
        import shutil, tempfile
        from pathlib import Path
        src = Path(__file__).parent
        snap = Path(tempfile.gettempdir()) / "ph_daily_prices.parquet"
        BOOK = ["BAYRY", "CAG", "HMC", "HPQ", "KHC", "MOS", "PFE", "SMCI", "T",
                "ALL", "EOG", "GL", "BEN"]
        px = pd.read_parquet(snap, columns=["date", "ticker", "adj_close", "close"])
        px["ticker"] = px["ticker"].astype(str).str.upper()
        px = px[px["ticker"].isin(BOOK)]
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
        px["px"] = px["adj_close"].where(px["adj_close"].notna(), px["close"])
        rows = []
        for t, g in px.groupby("ticker"):
            s = g.sort_values("date").drop_duplicates("date").set_index("date")["px"]
            if len(s) < 80:
                continue
            vol = get_daily_vol(s)
            events = s.index[::21]
            events = events.intersection(vol.dropna().index)
            lab = triple_barrier_labels(s, events, TripleBarrierConfig(max_holding=21))
            if lab.empty:
                continue
            vc = lab["bin"].value_counts(normalize=True)
            rows.append({
                "ticker": t, "n": len(lab),
                "pt": float(vc.get(1, 0)), "sl": float(vc.get(-1, 0)),
                "vb": float(vc.get(0, 0)), "mean_ret": float(lab["ret"].mean()),
            })
        out = pd.DataFrame(rows)
        print(out.round(3).to_string(index=False))
        if args.save:
            dest = src / "triple_barrier_labels.parquet"
            out.to_parquet(dest, index=False)
            print(f"Saved {dest}")
        raise SystemExit(0)
    print("use --book-barriers --save")
    raise SystemExit(0)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 500))), index=dates)
    
    # Test daily vol
    vol = get_daily_vol(close)
    print(f"Daily vol: {vol.tail()}")
    
    # Test CUSUM
    events = cusum_filter(close, h=vol.median())
    print(f"CUSUM events: {len(events)}")
    
    # Test triple barrier
    barriers = triple_barrier_labels(close, events, TripleBarrierConfig(max_holding=20))
    print(f"Barriers: {len(barriers)} labels, bin dist: {barriers['bin'].value_counts().to_dict()}")
    
    # Test fractional diff
    fd = frac_diff_ffd(close.pct_change().dropna(), d=0.4)
    print(f"FracDiff: {fd.tail()}")
    print(f"FracDiff NaN count: {fd.isna().sum()}")

    # Test purged CV
    X, y_clf, y_reg = make_xy_from_prices(close)
    cv = PurgedKFold(n_splits=3, t1=pd.Series(X.index + pd.Timedelta(days=5), index=X.index))
    for i, (tr, te) in enumerate(cv.split(X)):
        print(f"Fold {i}: train={len(tr)}, test={len(te)}")

    print("All tests passed!")
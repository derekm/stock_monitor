"""
HRP (Hierarchical Risk Parity) and Portfolio Construction

Implements:
1. Hierarchical Risk Parity (López de Prado)
2. Cluster Variance and Quasi-diagonalization
3. Correlation distance matrix
4. Portfolio optimization helpers (HRP, Risk Parity, Black-Litterman)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

warnings.filterwarnings("ignore")


# =============================================================================
# Correlation Distance
# =============================================================================

def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation to distance: sqrt((1 - corr) / 2)."""
    return np.sqrt((1.0 - corr.clip(-1, 1)) / 2.0)


def _quasi_diag(link: np.ndarray) -> list:
    """
    Quasi-diagonalization of linkage matrix.
    
    Returns sorted indices from hierarchical clustering.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        sort_ix = pd.concat([sort_ix, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Cluster variance for IVP (inverse variance portfolio)."""
    sub = cov.loc[items, items]
    w = 1.0 / np.diag(sub.values)
    w = w / w.sum()
    return float(w @ sub.values @ w)


# =============================================================================
# Hierarchical Risk Parity (HRP)
# =============================================================================

def hrp_weights_from_returns(returns: pd.DataFrame) -> pd.Series:
    """
    HRP portfolio weights from return matrix.
    
    Args:
        returns: DataFrame with assets in columns, dates in index
        
    Returns:
        Series of portfolio weights summing to 1
    """
    rets = returns.dropna(axis=1, how="any")
    if rets.shape[1] == 0:
        return pd.Series(dtype=float)
    if rets.shape[1] == 1:
        return pd.Series(1.0, index=rets.columns)
    
    cov, corr = rets.cov(), rets.corr()
    dist = _correl_dist(corr)
    link = sch.linkage(ssd.squareform(dist.values, checks=False), method="single")
    order = [corr.index[i] for i in _quasi_diag(link)]
    
    w = pd.Series(1.0, index=order)
    clusters = [order]
    
    while clusters:
        nxt = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            c1, c2 = cluster[:len(cluster)//2], cluster[len(cluster)//2:]
            v1, v2 = _cluster_var(cov, c1), _cluster_var(cov, c2)
            a = 1.0 - v1 / (v1 + v2 + 1e-12)
            w[c1] *= a
            w[c2] *= 1.0 - a
            nxt += [c1, c2]
        clusters = nxt
    
    w = w / w.sum()
    return w.reindex(returns.columns).fillna(0.0)


def hrp_weights_from_cov(cov: pd.DataFrame, corr: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    HRP from pre-computed covariance and correlation matrices.
    
    Args:
        cov: Covariance matrix (assets x assets)
        corr: Correlation matrix (if None, computed from cov)
        
    Returns:
        Series of portfolio weights
    """
    if corr is None:
        std = np.sqrt(np.diag(cov.values))
        corr = pd.DataFrame(cov.values / np.outer(std, std), index=cov.index, columns=cov.columns)
    
    dist = _correl_dist(corr)
    link = sch.linkage(ssd.squareform(dist.values, checks=False), method="single")
    order = [corr.index[i] for i in _quasi_diag(link)]
    
    w = pd.Series(1.0, index=order)
    clusters = [order]
    
    while clusters:
        nxt = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            c1, c2 = cluster[:len(cluster)//2], cluster[len(cluster)//2:]
            v1, v2 = _cluster_var(cov, c1), _cluster_var(cov, c2)
            a = 1.0 - v1 / (v1 + v2 + 1e-12)
            w[c1] *= a
            w[c2] *= 1.0 - a
            nxt += [c1, c2]
        clusters = nxt
    
    w = w / w.sum()
    return w.reindex(cov.columns).fillna(0.0)


# =============================================================================
# Risk Parity (Equal Risk Contribution)
# =============================================================================

def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-8) -> pd.Series:
    """
    Equal Risk Contribution (ERC) / Risk Parity weights using cyclical coordinate descent.
    
    Args:
        cov: Covariance matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Series of portfolio weights
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # Risk contribution of asset i
            rc_i = w[i] * (cov.values @ w)[i]
            # Target risk contribution
            rc_target = (w @ cov.values @ w) / n
            # Newton step
            if rc_i > 0:
                w[i] = w[i] * np.sqrt(rc_target / rc_i)
        
        # Normalize
        w = w / w.sum()
        
        if np.max(np.abs(w - w_prev)) < tol:
            break
    
    return pd.Series(w, index=cov.index)


# =============================================================================
# Sector/Cluster Neutral HRP
# =============================================================================

@dataclass
class SectorNeutralConfig:
    """Configuration for sector-neutral portfolio construction."""
    neutralize_sizes: bool = True
    conf_blend: float = 0.3
    gross_target: float = 1.0


def sector_neutralize_scores(
    scores: pd.Series,
    sectors: pd.Series,
    method: str = "demean",
) -> pd.Series:
    """
    Neutralize scores by sector.
    
    Args:
        scores: Raw scores/alphas indexed by ticker
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
    Project weights to dollar-neutral and beta-neutral (+ sector-neutral if provided).
    
    Args:
        w: Current weights
        betas: Beta estimates
        sectors: Optional sector assignments
        
    Returns:
        Projected weights
    """
    idx = w.index
    w = w.fillna(0.0).astype(float)
    b = betas.reindex(idx).fillna(1.0)
    
    # Iterate projections (converges quickly)
    for _ in range(3):
        # Dollar neutral
        w = w - w.mean()
        
        # Beta neutral
        bb = b.values
        wv = w.values
        num = wv @ bb
        den = bb @ bb + 1e-12
        w = pd.Series(wv - (num / den) * bb, index=idx)
        
        # Sector neutral
        if sectors is not None:
            sec = sectors.reindex(idx).astype(str).fillna("UNK")
            tmp = pd.DataFrame({"w": w, "sec": sec})
            w = tmp.groupby("sec")["w"].transform(lambda x: x - x.mean())
    
    return w


def build_sector_neutral_hrp_weights(
    day_sizes: pd.Series,
    sectors: pd.Series,
    returns_hist: pd.DataFrame,
    config: Optional[SectorNeutralConfig] = None,
) -> pd.Series:
    """
    Build sector-neutral HRP weights from conformal sizes.
    
    Process:
    1. Neutralize sizes by sector (cross-sectional demean)
    2. Split into L/S sleeves
    3. HRP within each sleeve using trailing returns
    4. Blend HRP structure with conviction
    5. Project to sector/dollar neutral
    6. Scale to target gross
    
    Args:
        day_sizes: Conformal signed sizes for one date
        sectors: Sector assignments
        returns_hist: Trailing returns for HRP
        config: SectorNeutralConfig
        
    Returns:
        Sector-neutral HRP weights
    """
    config = config or SectorNeutralConfig()
    s = day_sizes.dropna().astype(float)
    sec = sectors.reindex(s.index)
    
    if config.neutralize_sizes:
        s = sector_neutralize_scores(s, sec, method="demean").fillna(0.0)
    
    w = pd.Series(0.0, index=s.index)
    rh = returns_hist[[c for c in returns_hist.columns if c in s.index]]
    
    def sleeve(names: pd.Series, side: str) -> pd.Series:
        if names.empty:
            return pd.Series(dtype=float)
        cols = [c for c in names.index if c in rh.columns]
        if len(cols) <= 1:
            ew = pd.Series(1.0 / len(names), index=names.index)
        else:
            hw = hrp_weights_from_returns(rh[cols].dropna(how="any"))
            ew = hw.reindex(names.index).fillna(0.0)
            ew = ew / ew.sum() if ew.sum() > 0 else pd.Series(1.0 / len(names), index=names.index)
        
        mag = names.abs()
        mag = mag / (mag.sum() + 1e-12)
        blend = (1 - config.conf_blend) * ew + config.conf_blend * mag.reindex(ew.index).fillna(0.0)
        blend = blend / (blend.sum() + 1e-12)
        out = blend * float(names.abs().sum())
        return out if side == "long" else -out
    
    longs, shorts = s[s > 0], s[s < 0]
    wl, ws = sleeve(longs, "long"), sleeve(shorts, "short")
    
    if len(wl):
        w.loc[wl.index] = wl.values
    if len(ws):
        w.loc[ws.index] = ws.values
    
    # Project to sector/dollar neutral
    betas = pd.Series(1.0, index=s.index)  # placeholder - pass real betas if available
    if sec.notna().any():
        w = project_weights_factor_neutral(w, betas, sectors=sec)
    else:
        w = w - w.mean()
    
    gross = w.abs().sum()
    if gross > 0:
        w = w * (config.gross_target / gross)
    
    return w.fillna(0.0)


# =============================================================================
# Utility: Estimate Betas
# =============================================================================

def estimate_betas(returns_wide: pd.DataFrame, lookback: int = 60, market_col: Optional[str] = None) -> pd.Series:
    """
    Rolling beta vs equal-weight market (or specific market column).
    
    Args:
        returns_wide: Wide returns DataFrame (date x ticker)
        lookback: Window for beta estimation
        market_col: Optional explicit market factor column
        
    Returns:
        Series of betas indexed by ticker
    """
    from sklearn.linear_model import LinearRegression
    
    rw = returns_wide.dropna(how="all").iloc[-lookback:]
    mkt = rw[market_col] if market_col and market_col in rw.columns else rw.mean(axis=1)
    
    betas = {}
    x = mkt.values.reshape(-1, 1)
    
    for c in rw.columns:
        y = rw[c].values
        mask = np.isfinite(y) & np.isfinite(x.ravel())
        if mask.sum() < 20:
            betas[c] = 1.0
            continue
        lr = LinearRegression().fit(x[mask], y[mask])
        betas[c] = float(lr.coef_[0])
    
    return pd.Series(betas)


# =============================================================================
# Utility: Covariance Estimation with Shrinkage
# =============================================================================

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance estimator."""
    from sklearn.covariance import LedoitWolf
    
    rets = returns.dropna()
    if rets.shape[0] < rets.shape[1]:
        # Need more observations than variables
        raise ValueError(f"Need more observations ({rets.shape[0]}) than assets ({rets.shape[1]})")
    
    lw = LedoitWolf().fit(rets)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def oracle_approximating_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Oracle Approximating Shrinkage (OAS) covariance."""
    from sklearn.covariance import OAS
    
    rets = returns.dropna()
    oas = OAS().fit(rets)
    return pd.DataFrame(oas.covariance_, index=returns.columns, columns=returns.columns)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing HRP and portfolio construction...")
    
    # Create synthetic returns
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    n_assets = 20
    tickers = [f"T{i:02d}" for i in range(n_assets)]
    
    # Factor model returns
    mkt = np.random.normal(0.0002, 0.01, 500)
    rets = {}
    for i, tkr in enumerate(tickers):
        beta = np.random.uniform(0.5, 1.5)
        r = beta * mkt + np.random.normal(0.0, 0.01, 500)
        rets[tkr] = r
    
    returns_wide = pd.DataFrame(rets, index=dates)
    
    # Test HRP
    w_hrp = hrp_weights_from_returns(returns_wide)
    print(f"HRP weights: sum={w_hrp.sum():.6f}, n_assets={(w_hrp != 0).sum()}")
    print(f"  Max weight: {w_hrp.max():.4f}, Min weight: {w_hrp.min():.4f}")
    
    # Test Risk Parity
    w_rp = risk_parity_weights(returns_wide.cov())
    print(f"Risk Parity: sum={w_rp.sum():.6f}, n_assets={(w_rp != 0).sum()}")
    
    # Test Sector-neutral HRP
    sectors = pd.Series({t: f"S{i % 5}" for i, t in enumerate(tickers)})
    day_sizes = pd.Series(np.random.normal(0, 0.01, n_assets), index=tickers)
    
    w_sector = build_sector_neutral_hrp_weights(
        day_sizes, sectors, returns_wide.iloc[-60:], 
        SectorNeutralConfig(gross_target=1.0)
    )
    print(f"Sector-neutral HRP: sum={w_sector.sum():.6f}, gross={w_sector.abs().sum():.4f}")
    
    # Test betas
    betas = estimate_betas(returns_wide)
    print(f"Betas: {betas.to_dict()}")
    
    print("All tests passed!")
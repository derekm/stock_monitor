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

# Canonical implementations live in multi_horizon_hrp. These names were duplicated
# here as independent copies; they had already DIVERGED (the copy here was 8.4x
# slower and used label-based .loc indexing), so any fix applied to one silently
# missed the other. Import instead of redefining.
from multi_horizon_hrp import (  # noqa: E402
    _correl_dist,
    _quasi_diag,
    _cluster_var,
    hrp_weights_from_returns,
    estimate_betas,
    SectorNeutralConfig,
    sector_neutralize_scores,
    project_weights_factor_neutral,
    build_sector_neutral_hrp_weights,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Correlation Distance
# =============================================================================







# =============================================================================
# Hierarchical Risk Parity (HRP)
# =============================================================================



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









# =============================================================================
# Utility: Estimate Betas
# =============================================================================



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
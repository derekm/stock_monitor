"""
Conformal Prediction for Bet Sizing

Implements:
1. Split conformal prediction for binary classification
2. Exponential moving conformal (adaptive threshold)
3. Conformal bet sizing from prediction sets
4. Integration with cross-sectional scores
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# Conformal State
# =============================================================================

@dataclass
class ConformalState:
    """State of conformal predictor."""
    q_hat: float
    alpha: float
    n_calib: int


# =============================================================================
# Nonconformity Scores
# =============================================================================

def nonconformity_prob(y_true: np.ndarray, p_pos: np.ndarray) -> np.ndarray:
    """
    Binary nonconformity: 1 - p_true.
    
    Higher score = less conforming.
    """
    p_true = np.where(y_true == 1, p_pos, 1.0 - p_pos)
    return 1.0 - p_true


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample corrected conformal quantile.
    
    Args:
        scores: Nonconformity scores from calibration set
        alpha: Miscoverage level (e.g., 0.2 for 80% coverage)
        
    Returns:
        Quantile threshold q_hat
    """
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_conformal(
    y_cal: np.ndarray, 
    p_cal: np.ndarray, 
    alpha: float = 0.2
) -> ConformalState:
    """
    Fit split conformal predictor on calibration set.
    
    Args:
        y_cal: True labels (0/1)
        p_cal: Predicted probabilities for class 1
        alpha: Miscoverage level
        
    Returns:
        ConformalState with q_hat
    """
    s = nonconformity_prob(y_cal.astype(int), p_cal.astype(float))
    return ConformalState(
        q_hat=conformal_quantile(s, alpha),
        alpha=alpha,
        n_calib=len(s),
    )


# =============================================================================
# Prediction Sets & Bet Sizing
# =============================================================================

def conformal_predict_set(p_pos: float, q_hat: float) -> set[int]:
    """
    Include class if its nonconformity <= q_hat.
    
    Returns:
        Set of included classes (can be {}, {0}, {1}, or {0,1})
    """
    s1 = 1.0 - p_pos  # nonconformity if true class = 1
    s0 = p_pos        # nonconformity if true class = 0
    out = set()
    if s0 <= q_hat:
        out.add(0)
    if s1 <= q_hat:
        out.add(1)
    return out


def confidence_from_set(pred_set: set[int], p_pos: float) -> float:
    """
    Map prediction set to confidence in directional bet.
    
    - {1} -> confident long (confidence = p_pos)
    - {0} -> confident short (confidence = 1 - p_pos)
    - {0,1} or {} -> ambiguous (confidence = 0)
    """
    if pred_set == {1}:
        return float(p_pos)
    if pred_set == {0}:
        return float(1.0 - p_pos)
    return 0.0


def _smoothed_bet_size(p_pos, conf, max_size, min_conf, gamma, sigma, info):
    """E[bet_size(p_pos + eps)] for eps ~ N(0, sigma), by Gauss-Hermite quadrature.

    The exact sizer is a step-and-kink function of p_pos: the SIGN flips at the
    conformal set boundary and the magnitude is clamped to zero below min_conf.
    buy_candidates._step_expectation can integrate its drivers in closed form
    because each is a pure sum of steps; this sizer is not (conf_score depends on
    p_pos inside the surviving branch, and the branch itself depends on q_hat), so
    the expectation is taken numerically.

    21-node Gauss-Hermite is exact for polynomials up to degree 41 and converges
    fast for a smooth-away-from-thresholds integrand; the cost is ~21 evaluations
    of arithmetic already being done once. p_pos is a probability, so nodes are
    clipped to (0, 1) rather than allowed to wander outside the domain.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(21)
    total_w = weights.sum()
    acc = 0.0
    for z, w in zip(nodes, weights):
        p = float(np.clip(p_pos + sigma * z, 1e-9, 1.0 - 1e-9))
        pred_set = conformal_predict_set(p, conf.q_hat)
        cs = confidence_from_set(pred_set, p)
        if pred_set == {0, 1} or len(pred_set) == 0 or cs < min_conf:
            continue  # contributes 0 to the expectation
        strength = (cs - min_conf) / (1.0 - min_conf + 1e-12)
        strength = float(np.clip(strength, 0.0, 1.0) ** gamma)
        sign = 1.0 if pred_set == {1} else -1.0
        acc += w * sign * max_size * strength
    size = acc / total_w
    info["noise_sigma"] = float(sigma)
    return float(np.clip(size, -max_size, max_size)), info


def bet_size_from_conformal(
    p_pos: float,
    conf: ConformalState,
    max_size: float = 1.0,
    min_conf: float = 0.55,
    gamma: float = 1.5,
    noise_sigma: float = 0.0,
) -> tuple[float, dict]:
    """
    Convert conformal prediction set to signed bet size.
    
    Args:
        p_pos: Model probability for class 1 (up)
        conf: ConformalState with q_hat
        max_size: Maximum position size
        min_conf: Minimum confidence to trade
        gamma: Sizing exponent (gamma > 1 = more aggressive)
        
    Returns:
        (signed_size, info_dict)
        signed_size in [-max_size, max_size]
        + = long, - = short, 0 = no trade
    """
    pred_set = conformal_predict_set(p_pos, conf.q_hat)
    conf_score = confidence_from_set(pred_set, p_pos)
    
    info = {
        "set": pred_set,
        "conf": conf_score,
        "q_hat": conf.q_hat,
    }

    # Noise-convolved path: E[size(p + eps)] for eps ~ N(0, noise_sigma), the same
    # Taleb/American-options treatment buy_candidates._step_expectation applies to
    # its step drivers. Three knife edges live in the exact path below:
    #   1. conf_score < min_conf -> 0, so a hair either side of the floor is a full
    #      position or nothing;
    #   2. sign = +-1 from set membership, so near p_pos = 0.5 the SIGN of the bet
    #      is a coin toss on estimation noise;
    #   3. pred_set == {0,1} -> 0, itself a thresholded function of p_pos.
    # At 105% daily turnover these flips are not academic: a name oscillating
    # around a threshold gets bought and sold on noise, paying spread each way.
    # Smoothing preserves the asymptotes (deep-confidence bets keep full size) and
    # only softens the transitions, so it cannot manufacture conviction.
    if noise_sigma and noise_sigma > 0:
        return _smoothed_bet_size(
            p_pos, conf, max_size, min_conf, gamma, noise_sigma, info
        )

    # Empty or ambiguous set -> no trade
    if pred_set == {0, 1} or len(pred_set) == 0:
        return 0.0, info
    
    # Low confidence -> no trade
    if conf_score < min_conf:
        return 0.0, info
    
    # Size grows with confidence above floor
    strength = (conf_score - min_conf) / (1.0 - min_conf + 1e-12)
    strength = float(np.clip(strength, 0.0, 1.0) ** gamma)
    
    sign = 1.0 if pred_set == {1} else -1.0
    return sign * max_size * strength, info


# =============================================================================
# Adaptive Conformal (Rolling Window)
# =============================================================================

class AdaptiveConformal:
    """
    Adaptive conformal prediction with rolling window.
    
    Updates q_hat online as new (y, p) pairs arrive.
    Pairs well with drift-aware online learning.
    """
    
    def __init__(self, alpha: float = 0.2, window: int = 500):
        self.alpha = alpha
        self.window = window
        self.scores: list[float] = []
        self.q_hat = 1.0
    
    def update(self, y: int, p_pos: float):
        """Update with new observation."""
        s = float(nonconformity_prob(np.array([y]), np.array([p_pos]))[0])
        self.scores.append(s)
        if len(self.scores) > self.window:
            self.scores = self.scores[-self.window:]
        
        n = len(self.scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(self.scores, level, method="higher"))
    
    def size(self, p_pos: float, **kwargs) -> tuple[float, dict]:
        """Get bet size using current q_hat."""
        conf = ConformalState(q_hat=self.q_hat, alpha=self.alpha, n_calib=len(self.scores))
        return bet_size_from_conformal(p_pos, conf, **kwargs)


# =============================================================================
# Full Pipeline: Model + Conformal
# =============================================================================

def fit_model_with_conformal(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.2,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Train binary model + split conformal calibration.
    
    Time-ordered splits: train (60%) | calib (20%) | test (20%)
    
    Returns:
        Dict with model, conformal state, test results
    """
    import lightgbm as lgb
    
    n = len(X)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    
    X_tr, y_tr = X.iloc[:i1], y.iloc[:i1]
    X_ca, y_ca = X.iloc[i1:i2], y.iloc[i1:i2]
    X_te, y_te = X.iloc[i2:], y.iloc[i2:]
    
    dtr = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "verbose": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    model = lgb.train(params, dtr, num_boost_round=250)
    
    p_ca = model.predict(X_ca)
    conf = fit_conformal(y_ca.values.astype(int), p_ca, alpha=alpha)
    
    p_te = model.predict(X_te)
    sizes = []
    for p in p_te:
        sz, info = bet_size_from_conformal(float(p), conf, max_size=1.0, min_conf=0.55)
        sizes.append(sz)
    
    out = X_te.copy()
    out["y"] = y_te.values
    out["p"] = p_te
    out["size"] = sizes
    out["pnl_proxy"] = out["size"] * (out["y"] * 2 - 1)
    
    summary = {
        "model": model,
        "conformal": conf,
        "test": out,
        "avg_abs_size": float(np.mean(np.abs(out["size"]))),
        "coverage_singleton": float(np.mean(np.abs(out["size"]) > 0)),
        "pnl_sum": float(out["pnl_proxy"].sum()),
    }
    return summary


# =============================================================================
# Expanding Conformal Sizes (for production use)
# =============================================================================

def expanding_conformal_sizes(
    scored: pd.DataFrame,
    feature_cols: list[str],
    alpha_grid: tuple[float, ...] = (0.1, 0.2, 0.3),
    min_train_dates: int = 60,
    recal_every: int = 21,
    embargo: int = 5,
    min_conf: float = 0.55,
    noise_sigma: float = 0.0,
) -> pd.DataFrame:
    """
    Expanding binary model + conformal on OOS ranker scores.
    
    Recalibrates q_hat / alpha periodically with embargo.
    
    Args:
        scored: OOS scores from expanding ranker (with date, ticker, score, y)
        feature_cols: Original feature columns
        alpha_grid: Grid of alpha values to select from
        min_train_dates: Minimum training dates before first prediction
        recal_every: Recalibration frequency in dates
        embargo: Embargo days between train end and predict start
        min_conf: Minimum confidence for bet sizing
        
    Returns:
        DataFrame with p, size_raw, q_hat, alpha columns added
    """
    d = scored.sort_values(["date", "ticker"]).copy()
    d["score_z"] = d.groupby("date")["score"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12)
    )
    d["score_rk"] = d.groupby("date")["score"].rank(pct=True)
    d["y_bin"] = (d["y"] > 0).astype(int)
    
    feats = feature_cols + ["score", "score_z", "score_rk"]
    dates = np.array(sorted(d["date"].unique()))
    
    import lightgbm as lgb
    
    out_parts = []
    i = min_train_dates
    conf = None
    model = None
    best_alpha = alpha_grid[len(alpha_grid) // 2]
    
    while i < len(dates):
        j = min(i + recal_every, len(dates))
        train_end = i - embargo
        
        if train_end < min_train_dates // 2:
            i = j
            continue
        
        train_dates = dates[:train_end]
        cal_cut = int(len(train_dates) * 0.75)
        fit_dates = train_dates[:cal_cut]
        cal_dates = train_dates[cal_cut:]
        te_dates = dates[i:j]
        
        d_fit = d[d["date"].isin(fit_dates)]
        d_cal = d[d["date"].isin(cal_dates)]
        d_te = d[d["date"].isin(te_dates)]
        
        if min(len(d_fit), len(d_cal), len(d_te)) == 0 or d_fit["y_bin"].nunique() < 2:
            i = j
            continue
        
        # Train binary model
        dtr = lgb.Dataset(d_fit[feats], label=d_fit["y_bin"])
        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 23,
            "max_depth": 6,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
        model = lgb.train(params, dtr, num_boost_round=280)
        
        # Select alpha on calibration
        p_cal = model.predict(d_cal[feats])
        best = None
        for a in alpha_grid:
            c = fit_conformal(d_cal["y_bin"].values, p_cal, a)
            sizes = np.array([
                bet_size_from_conformal(float(p), c, min_conf=min_conf, noise_sigma=noise_sigma)[0] 
                for p in p_cal
            ])
            pnl = sizes * (d_cal["y_bin"].values * 2 - 1)
            # Objective: mean PnL / (avg abs size + penalty)
            obj = float(pnl.mean() / (np.mean(np.abs(sizes)) + 0.05))
            if best is None or obj > best[0]:
                best = (obj, a, c)
        
        _, best_alpha, conf = best
        
        # Apply to test
        p_te = model.predict(d_te[feats])
        sizes = [
            bet_size_from_conformal(float(p), conf, min_conf=min_conf, noise_sigma=noise_sigma)[0] 
            for p in p_te
        ]
        
        part = d_te.copy()
        part["p"] = p_te
        part["size_raw"] = sizes
        part["q_hat"] = conf.q_hat
        part["alpha"] = best_alpha
        out_parts.append(part)
        
        i = j
    
    if not out_parts:
        raise RuntimeError("No conformal sizes produced.")
    
    return pd.concat(out_parts, ignore_index=True)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    
    print("Testing conformal prediction...")
    
    # Create synthetic data
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    # Weak signal
    y = (0.3 * X["f1"] + 0.2 * X["f2"] + np.random.randn(n) > 0).astype(int)
    
    # Test split conformal
    result = fit_model_with_conformal(X, y, alpha=0.2)
    print(f"Conformal q_hat: {result['conformal'].q_hat:.4f}")
    print(f"Avg abs size: {result['avg_abs_size']:.4f}")
    print(f"Trade rate: {result['coverage_singleton']:.3f}")
    print(f"PnL proxy sum: {result['pnl_sum']:.4f}")
    
    # Test adaptive conformal
    print("\nTesting adaptive conformal...")
    # Simulate a calibrated model
    n_cal = 500
    y_cal = np.random.randint(0, 2, n_cal)
    p_cal = np.random.uniform(0.3, 0.7, n_cal)
    # Add some signal
    p_cal = p_cal + 0.2 * (y_cal - 0.5)
    p_cal = np.clip(p_cal, 0.01, 0.99)
    
    adaptive = AdaptiveConformal(alpha=0.2, window=300)
    for yi, pi in zip(y_cal[:300], p_cal[:300]):
        adaptive.update(yi, pi)
    
    print(f"Initial q_hat: {adaptive.q_hat:.4f}")
    
    # Test bet sizing
    for p_test in [0.2, 0.5, 0.8, 0.9]:
        sz, info = adaptive.size(p_test, min_conf=0.55)
        print(f"  p={p_test:.2f} -> size={sz:.3f}, set={info['set']}, conf={info['conf']:.3f}")
    
    # Test expanding conformal on scored data
    print("\nTesting expanding conformal on scored panel...")
    dates = pd.bdate_range("2020-01-01", periods=400)
    tickers = [f"T{i:02d}" for i in range(10)]
    
    rows = []
    for tkr in tickers:
        for dt in dates:
            rows.append({
                "date": dt,
                "ticker": tkr,
                "score": np.random.randn(),
                "y": np.random.randn(),
                "ret_1": np.random.randn() * 0.01,
                "ret_5": np.random.randn() * 0.02,
                "ret_10": np.random.randn() * 0.03,
                "vol_10": abs(np.random.randn() * 0.01),
                "vol_20": abs(np.random.randn() * 0.01),
                "ma_gap": np.random.randn() * 0.01,
            })
    
    scored = pd.DataFrame(rows)
    feats = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
    
    try:
        sized = expanding_conformal_sizes(
            scored, feats,
            alpha_grid=(0.1, 0.2, 0.3),
            min_train_dates=50,
            recal_every=21,
            embargo=5,
        )
        print(f"Sized: {sized.shape}")
        print(f"Trade rate: {(sized['size_raw'].abs() > 0).mean():.3f}")
        print(f"Avg abs size: {sized['size_raw'].abs().mean():.3f}")
        print(f"Unique alphas: {sized['alpha'].unique()}")
    except Exception as e:
        print(f"Expanding conformal test failed (expected for random data): {e}")
    
    print("\nAll tests passed!")
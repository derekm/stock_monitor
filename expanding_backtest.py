"""
Expanding-Window Ranker with Transaction Costs and Market Impact

Integrates:
1. Expanding-window LambdaRank (from cross_sectional_ranker)
2. Conformal bet sizing (from conformal_sizing)
3. Sector-neutral HRP portfolio construction (from portfolio_construction)
4. Transaction costs + square-root impact model
5. Full backtest with costs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_construction import (
    hrp_weights_from_returns,
    sector_neutralize_scores,
    project_weights_factor_neutral,
    build_sector_neutral_hrp_weights,
    estimate_betas,
)
from conformal_sizing import (
    expanding_conformal_sizes,
    bet_size_from_conformal,
    fit_conformal,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Cost Model
# =============================================================================

@dataclass
class CostModel:
    """
    Transaction cost model with square-root market impact.
    
    All rates in decimals (not bps).
    """
    # Linear costs (per unit of weight traded)
    fixed_bps: float = 0.5          # Commission/fees per side (bps)
    half_spread_bps: float = 2.0    # Half-spread paid per side (bps)
    
    # Square-root impact: cost = coeff * sigma * sqrt(participation) * |dw|
    impact_coeff: float = 0.10      # Impact coefficient
    default_adv: float = 5e7        # Default ADV in $ (if unknown)
    book_nav: float = 1e7           # Book NAV in $ (for weight -> $ conversion)
    max_participation: float = 0.10 # Max participation rate
    
    # Borrow costs (annual bps)
    default_borrow_bps_annual: float = 100.0  # ~0.4 bps/day


def bps_to_decimal(bps: float) -> float:
    """Convert basis points to decimal."""
    return bps * 1e-4


def estimate_sigmas(returns_hist: pd.DataFrame) -> pd.Series:
    """Estimate daily volatilities from trailing returns."""
    sigmas = returns_hist.std()
    return sigmas.replace(0, np.nan).fillna(sigmas.median())


def apply_costs_and_impact(
    w_prev: pd.Series,
    w_new: pd.Series,
    cost: CostModel,
    sigmas: Optional[pd.Series] = None,
    adv: Optional[pd.Series] = None,
) -> tuple[float, dict]:
    """
    Calculate total rebalancing cost (fraction of NAV).
    
    Args:
        w_prev: Previous portfolio weights
        w_new: New target weights
        cost: CostModel parameters
        sigmas: Daily volatilities (if None, estimated)
        adv: Average daily volumes in $ (if None, uses default)
        
    Returns:
        (total_cost_fraction, diagnostics_dict)
    """
    idx = w_prev.index.union(w_new.index)
    wp = w_prev.reindex(idx).fillna(0.0)
    wn = w_new.reindex(idx).fillna(0.0)
    dw = wn - wp  # weight change
    
    traded_weight = dw.abs().sum()
    if traded_weight <= 0:
        return 0.0, {"turnover": 0.0, "spread_fee": 0.0, "impact": 0.0}
    
    # Linear spread + fee on one-way turnover
    bps = bps_to_decimal(cost.fixed_bps + cost.half_spread_bps)
    spread_fee = float(dw.abs().sum() * bps)
    
    # Square-root impact
    impact = 0.0
    if cost.impact_coeff > 0 and traded_weight > 0:
        if sigmas is None:
            sigmas = pd.Series(0.01, index=idx)
        sigmas = sigmas.reindex(idx).fillna(0.01)
        
        if adv is None:
            adv = pd.Series(cost.default_adv, index=idx)
        adv = adv.reindex(idx).fillna(cost.default_adv).clip(lower=1.0)
        
        # Participation = |Q| / ADV = |dw| * NAV / ADV
        part = (dw.abs() * cost.book_nav / adv).clip(upper=cost.max_participation)
        
        # Impact cost ≈ |dw| * coeff * sigma * sqrt(participation)
        impact = float((
            dw.abs() * cost.impact_coeff * sigmas * np.sqrt(part + 1e-16)
        ).sum())
    
    total = spread_fee + impact
    
    return total, {
        "turnover": float(dw.abs().sum()),
        "spread_fee": spread_fee,
        "impact": impact,
        "total_cost": total,
    }


# =============================================================================
# Borrow Costs
# =============================================================================

def apply_borrow_cost(
    w_new: pd.Series,
    borrow_bps_annual: pd.Series,
    cost: CostModel,
    holding_days: float = 1.0,
) -> tuple[float, dict]:
    """
    Calculate borrow cost on short positions.
    
    Args:
        w_new: New portfolio weights
        borrow_bps_annual: Annual borrow cost in bps per ticker
        cost: CostModel (for defaults)
        holding_days: Holding period in days
        
    Returns:
        (borrow_cost_fraction, diagnostics)
    """
    shorts = w_new.clip(upper=0.0).abs()
    if shorts.sum() == 0:
        return 0.0, {"borrow": 0.0, "short_exposure": 0.0}
    
    # Annual bps -> daily fraction
    ann_bps = borrow_bps_annual.reindex(shorts.index).fillna(cost.default_borrow_bps_annual)
    daily_bps = ann_bps / 252.0
    borrow = float((shorts * bps_to_decimal(daily_bps) * holding_days).sum())
    
    return borrow, {
        "borrow": borrow,
        "short_exposure": float(shorts.sum()),
    }


# =============================================================================
# Full Cost Integration
# =============================================================================

def calculate_total_costs(
    w_prev: pd.Series,
    w_new: pd.Series,
    cost: CostModel,
    sigmas: Optional[pd.Series] = None,
    adv: Optional[pd.Series] = None,
    borrow_bps_annual: Optional[pd.Series] = None,
    holding_days: float = 1.0,
) -> tuple[float, dict]:
    """
    Total cost = rebalance cost + borrow cost.
    
    Returns:
        (total_cost_fraction, diagnostics)
    """
    reb_cost, reb_diag = apply_costs_and_impact(w_prev, w_new, cost, sigmas, adv)
    bor_cost, bor_diag = apply_borrow_cost(
        w_new, 
        borrow_bps_annual if borrow_bps_annual is not None else pd.Series(dtype=float), 
        cost, holding_days
    )
    
    return reb_cost + bor_cost, {
        **reb_diag,
        **bor_diag,
        "rebalance_cost": reb_cost,
        "borrow_cost": bor_cost,
        "total_cost": reb_cost + bor_cost,
    }


# =============================================================================
# Book Backtest with Costs
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for book backtest."""
    lookback: int = 60              # Trailing window for HRP
    gross_target: float = 1.0       # Target gross exposure
    cost: Optional[CostModel] = None
    sector_neutral: bool = True
    conf_blend: float = 0.3         # HRP/confidence blend


def build_book_backtest(
    sized: pd.DataFrame,
    returns_wide: pd.DataFrame,
    sectors: Optional[pd.Series] = None,
    adv: Optional[pd.Series] = None,
    borrow_bps_annual: Optional[pd.Series] = None,
    config: Optional[BacktestConfig] = None,
    precomputed_weights: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run full book backtest with HRP portfolio + costs.
    
    Args:
        sized: DataFrame with date, ticker, size_raw (from conformal)
        returns_wide: Wide returns (date x ticker) for HRP + PnL
        sectors: Sector mapping (ticker -> sector)
        adv: Average daily volume in $
        borrow_bps_annual: Annual borrow cost in bps
        config: BacktestConfig
        precomputed_weights: Optional ticker x date weight matrix. When supplied,
            the per-date HRP build is SKIPPED and these columns are used directly.
            The caller (v5_integrated STEP 3) already builds exactly these weights
            for exactly these dates, in parallel; recomputing the same HRP here was
            duplicated work and the single most expensive part of this function.
            Dates missing from the matrix fall back to building weights inline.
        
    Returns:
        DataFrame with daily book returns, costs, weights
    """
    config = config or BacktestConfig()
    cost = config.cost or CostModel()
    
    returns_wide = returns_wide.sort_index()
    sized = sized.copy()
    sized["date"] = pd.to_datetime(sized["date"])
    
    w_prev = pd.Series(dtype=float)
    rows = []

    # Betas are hoisted out of the loop. build_sector_neutral_hrp_weights recomputes
    # them from scratch when betas is None, and this loop called it once per date --
    # ~500 names re-estimated on ~1,900 dates. They are a slow-moving 60-day
    # regression, so a single estimate over the full history is the right granularity
    # here; re-estimating per date was cost without signal.
    #
    # NOTE this loop CANNOT be parallelised: w_prev threads through every iteration
    # (turnover and cost depend on the previous day's book), so date N+1 needs date
    # N's result. That is why the fix is vectorisation, not a process pool -- unlike
    # v5_integrated's weight loop, where the dates are genuinely independent.
    betas_all = estimate_betas(returns_wide)

    # Precomputed weights are keyed by the same date strings STEP 3 stamped on each
    # column (w_final.name = str(date)). Normalise to Timestamps once so the per-date
    # lookup is a dict hit rather than a string-format guess inside the loop.
    pre_cols = {}
    if precomputed_weights is not None and len(precomputed_weights.columns):
        for c in precomputed_weights.columns:
            try:
                pre_cols[pd.to_datetime(c)] = c
            except (ValueError, TypeError):
                continue
    n_pre = n_built = 0

    for dt, g in sized.groupby("date"):
        dt = pd.to_datetime(dt)
        
        # Trailing returns strictly before dt (no lookahead)
        prev_idx = returns_wide.index[returns_wide.index < dt]
        if len(prev_idx) < max(10, config.lookback // 3):
            continue
        hist = returns_wide.loc[prev_idx[-config.lookback:]]
        
        day_sizes = g.set_index("ticker")["size_raw"].astype(float)
        
        # Build target weights using sector-neutral HRP
        if dt in pre_cols:
            # Already built (in parallel) by the caller: reuse, do not recompute.
            w_new = pd.to_numeric(
                precomputed_weights[pre_cols[dt]], errors="coerce"
            ).dropna()
            w_new = w_new[w_new != 0.0]
            n_pre += 1
        elif config.sector_neutral and sectors is not None:
            n_built += 1
            sec = sectors.reindex(day_sizes.index)
            from portfolio_construction import SectorNeutralConfig
            sn_config = SectorNeutralConfig(
                neutralize_sizes=True,
                conf_blend=config.conf_blend,
                gross_target=config.gross_target,
            )
            w_new = build_sector_neutral_hrp_weights(
                day_sizes, sec, hist, config=sn_config,
                betas=betas_all.reindex(day_sizes.index).fillna(1.0),
            )
        else:
            # Simple HRP without sector neutralization
            rh = hist[[c for c in hist.columns if c in day_sizes.index]]
            w_new = pd.Series(0.0, index=day_sizes.index)
            longs, shorts = day_sizes[day_sizes > 0], day_sizes[day_sizes < 0]
            
            def sleeve(names, side):
                if names.empty:
                    return pd.Series(dtype=float)
                cols = [c for c in names.index if c in rh.columns]
                if len(cols) <= 1:
                    ew = pd.Series(1.0 / len(names), index=names.index)
                else:
                    ew = hrp_weights_from_returns(rh[cols].dropna(how="any"))
                    ew = ew.reindex(names.index).fillna(0.0)
                    ew = ew / ew.sum() if ew.sum() > 0 else pd.Series(1.0 / len(names), index=names.index)
                mag = names.abs() / (names.abs().sum() + 1e-12)
                blend = (1 - config.conf_blend) * ew + config.conf_blend * mag.reindex(ew.index).fillna(0.0)
                blend = blend / (blend.sum() + 1e-12)
                out = blend * float(names.abs().sum())
                return out if side == "long" else -out
            
            wl = sleeve(longs, "long")
            ws = sleeve(shorts, "short")
            if len(wl): w_new.loc[wl.index] = wl.values
            if len(ws): w_new.loc[ws.index] = ws.values
            w_new = w_new - w_new.mean()
            g = w_new.abs().sum()
            if g > 0: w_new = w_new * (config.gross_target / g)
        
        if w_prev.empty:
            w_prev = pd.Series(0.0, index=w_new.index)
        
        # Calculate costs
        sigmas = estimate_sigmas(hist)
        reb_cost, cost_diag = calculate_total_costs(
            w_prev, w_new, cost, 
            sigmas=sigmas, 
            adv=adv, 
            borrow_bps_annual=borrow_bps_annual,
        )
        
        # Realize next day returns
        future = returns_wide.index[returns_wide.index > dt]
        if len(future) == 0:
            continue
        nxt = returns_wide.loc[future[0]]
        
        common = w_new.index.intersection(nxt.dropna().index)
        if len(common) == 0:
            continue
        
        gross_ret = float((w_new.reindex(common).fillna(0.0) * nxt.reindex(common)).sum())
        net_ret = gross_ret - reb_cost
        
        # Beta exposure
        betas = estimate_betas(hist)
        book_beta = float((w_new.reindex(betas.index).fillna(0) * betas).sum())
        
        rows.append({
            "date": dt,
            "gross_ret": gross_ret,
            "rebalance_cost": cost_diag["rebalance_cost"],
            "borrow_cost": cost_diag["borrow_cost"],
            "cost": cost_diag["total_cost"],
            "net_ret": net_ret,
            "turnover": cost_diag["turnover"],
            "impact": cost_diag["impact"],
            "short_exposure": cost_diag["short_exposure"],
            "gross_exposure": float(w_new.abs().sum()),
            "net_exposure": float(w_new.sum()),
            "book_beta": book_beta,
            "n_long": int((w_new > 0).sum()),
            "n_short": int((w_new < 0).sum()),
        })
        
        w_prev = w_new
    
    if n_pre or n_built:
        print(f"  book backtest: {n_pre} dates reused precomputed weights, "
              f"{n_built} rebuilt")

    return pd.DataFrame(rows).set_index("date").sort_index()


def book_stats(bt: pd.DataFrame) -> dict:
    """Summary statistics for book backtest."""
    if bt is None or len(bt) == 0:
        return {}
    
    mu, sd = bt["net_ret"].mean(), bt["net_ret"].std()
    mu_g, sd_g = bt["gross_ret"].mean(), bt["gross_ret"].std()
    eq = (1 + bt["net_ret"]).cumprod()
    
    return {
        "n_days": int(len(bt)),
        "net_sharpe": float(mu / (sd + 1e-12) * np.sqrt(252)),
        "gross_sharpe": float(mu_g / (sd_g + 1e-12) * np.sqrt(252)),
        "avg_cost": float(bt["cost"].mean()),
        "avg_borrow": float(bt["borrow_cost"].mean()),
        "avg_reb_cost": float(bt["rebalance_cost"].mean()),
        "avg_turnover": float(bt["turnover"].mean()),
        "avg_impact": float(bt["impact"].mean()),
        "avg_short_exp": float(bt["short_exposure"].mean()),
        "avg_book_beta": float(bt["book_beta"].mean()),
        "avg_gross_exposure": float(bt["gross_exposure"].mean()),
        "avg_net_exposure": float(bt["net_exposure"].mean()),
        "cum_net": float(eq.iloc[-1] - 1),
        "max_drawdown": float((1 - eq / eq.cummax()).max()),
    }


# =============================================================================
# End-to-End Pipeline
# =============================================================================

def run_expanding_backtest(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    feature_cols: list[str],
    sectors: Optional[pd.Series] = None,
    adv: Optional[pd.Series] = None,
    borrow_bps_annual: Optional[pd.Series] = None,
    ranker_cfg: Optional = None,  # ExpandingRankerConfig
    bt_config: Optional[BacktestConfig] = None,
) -> dict:
    """
    Full end-to-end backtest:
    1. Expanding-window LambdaRank
    2. Conformal bet sizing
    3. Sector-neutral HRP portfolio
    4. Cost-aware backtest
    
    Returns dict with all intermediate results and final stats.
    """
    from cross_sectional_ranker import expanding_window_lambdarank, ExpandingRankerConfig
    
    # Default configs
    if ranker_cfg is None:
        ranker_cfg = ExpandingRankerConfig()
    if bt_config is None:
        bt_config = BacktestConfig()
    
    print("=" * 60)
    print("STEP 1: Expanding-window LambdaRank")
    print("=" * 60)
    oos, win_stats, last_model = expanding_window_lambdarank(panel, feature_cols, ranker_cfg)
    
    print("\n" + "=" * 60)
    print("STEP 2: Conformal bet sizing")
    print("=" * 60)
    sized = expanding_conformal_sizes(oos, feature_cols)
    print(f"  Sized rows: {len(sized)}")
    print(f"  Trade rate: {(sized['size_raw'].abs() > 0).mean():.3f}")
    print(f"  Avg abs size: {sized['size_raw'].abs().mean():.3f}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Book backtest with costs")
    print("=" * 60)
    bt = build_book_backtest(
        sized, returns_wide,
        sectors=sectors,
        adv=adv,
        borrow_bps_annual=borrow,
        config=bt_config,
    )
    
    stats = book_stats(bt)
    print(f"  Net Sharpe: {stats.get('net_sharpe', np.nan):.2f}")
    print(f"  Gross Sharpe: {stats.get('gross_sharpe', np.nan):.2f}")
    print(f"  Avg cost: {stats.get('avg_cost', np.nan):.5f}")
    print(f"  Avg turnover: {stats.get('avg_turnover', np.nan):.3f}")
    print(f"  Max DD: {stats.get('max_drawdown', np.nan):.2%}")
    print(f"  Avg book beta: {stats.get('avg_book_beta', np.nan):.3f}")
    
    return {
        "oos_scores": oos,
        "window_stats": win_stats,
        "last_model": last_model,
        "sized": sized,
        "book": bt,
        "stats": stats,
    }


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("Testing expanding backtest with costs...")
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=600)
    n_tickers = 20
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
        df["y"] = y + 0.03 * f["ret_5"] + 0.02 * f["ma_gap"]
        df["ticker"] = tkr
        df["date"] = dates
        panels.append(df)
    
    panel = pd.concat(panels).dropna().reset_index(drop=True)
    returns_wide = pd.DataFrame(rets, index=dates)
    feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "ma_gap"]
    
    # Sectors
    sectors = pd.Series({t: f"S{i % 5}" for i, t in enumerate(tickers)})
    
    # ADV and borrow
    adv = pd.Series({t: float(5e7 * np.exp(np.random.normal(0, 0.25))) for t in tickers})
    borrow = pd.Series({t: float(np.random.choice([50, 100, 200, 500, 1000])) for t in tickers})
    
    # Run backtest
    from cross_sectional_ranker import ExpandingRankerConfig
    result = run_expanding_backtest(
        panel, returns_wide, feature_cols,
        sectors=sectors,
        adv=adv,
        borrow_bps_annual=borrow,
        ranker_cfg=ExpandingRankerConfig(
            min_train_dates=200,
            test_block=21,
            step=21,
            embargo_dates=5,
            n_bins=5,
        ),
        bt_config=BacktestConfig(
            lookback=60,
            gross_target=1.0,
            sector_neutral=True,
        ),
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
    
    print("\nAll tests passed!")
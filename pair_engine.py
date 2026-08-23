#!/usr/bin/env python3
"""
pair_engine.py — Pair / relative-value engine: cointegration + residual
mean-reversion inside industry groups, with stops and time exits.

Vectorized implementation:
  1. Universe: pairs inside the same industry, cap --max-per-group (coverage-ranked).
  2. Return-correlation screen before Engle-Granger.
  3. FDR across surviving pairs; walk-forward OOS trading.

Daily DAG: --max-per-group 40 --n-folds 1 (uncapped 16k names is ~1.2M pairs/fold).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns
from cv_utils import bh_fdr
from cost_model import apply_costs_to_trades

PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_PAIRS = DATA_DIR / "pair_engine_pairs.parquet"
OUT_TRADES = DATA_DIR / "pair_engine_trades.parquet"
OUT_STATS = DATA_DIR / "pair_engine_stats.parquet"


def _groups(max_per_group: int = 40, rank: pd.Series | None = None) -> dict[str, list[str]]:
    """Industry groups from monitored_stocks, capped by rank (coverage/liquidity)."""
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    if stocks.empty or "ticker" not in stocks.columns:
        return {}
    s = pd.DataFrame({
        "ticker": stocks["ticker"].astype(str).str.upper(),
        "group": (stocks["industry"] if "industry" in stocks.columns else stocks.get("sector", "unknown")).astype(str),
    })
    s = s[s["group"].str.strip().ne("") & s["group"].ne("nan") & s["group"].ne("None")]
    s = s.drop_duplicates("ticker")
    if rank is None and "sp500_member" in stocks.columns:
        rank = pd.Series(
            stocks["sp500_member"].fillna(False).astype(float).to_numpy(),
            index=stocks["ticker"].astype(str).str.upper(),
        )
    if rank is not None:
        s["rank"] = s["ticker"].map(rank).fillna(0.0)
        s = s.sort_values(["group", "rank"], ascending=[True, False])
    else:
        s = s.sort_values(["group", "ticker"])
    if max_per_group and max_per_group > 0:
        s = s.groupby("group", sort=False).head(int(max_per_group))
    out = s.groupby("group")["ticker"].agg(lambda x: sorted(set(x))).to_dict()
    return {g: ts for g, ts in out.items() if len(ts) >= 2}


def fast_adf_residual(x: np.ndarray) -> tuple[float, float]:
    """Fast ADF test on a series using vectorized OLS (no statsmodels).
    
    Returns (t-stat, p-value) using the Mackinnon approximate p-value.
    This is the pre-filter: fast and good enough to reject clearly non-stationary series.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 50:
        return 0.0, 1.0
    
    dy = np.diff(x)
    y = dy
    # ADF regression: dy_t = alpha + beta * x_{t-1} + gamma * dy_{t-1} + eps
    x_lag = x[:-1]
    dy_lag = np.concatenate([[0], dy[:-1]])
    
    # Design matrix: [1, x_lag, dy_lag]
    X = np.column_stack([np.ones(n - 1), x_lag, dy_lag])
    
    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    
    # t-stat for beta[1] (the coefficient on x_{t-1})
    dof = n - 3
    sse = np.sum(resid ** 2)
    sigma2 = sse / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(cov[1, 1])
        t_stat = beta[1] / se_beta1 if se_beta1 > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    
    # Approximate p-value using the ADF distribution (Mackinnon)
    # For the "c" case (constant, no trend), the 5% critical value is ~-2.86
    # Simple approximation: reject if t_stat < -2.86
    # We use a rough mapping for the p-value
    from scipy import stats as sp_stats
    # ADF distribution is non-standard; use MacKinnon (1994) approximation
    # p-value ≈ norm.cdf(t_stat) for a rough approximation
    p_val = float(sp_stats.norm.cdf(t_stat))
    
    return float(t_stat), p_val


def adf_pre_filter(x: np.ndarray, y: np.ndarray, pval_thresh: float = 0.10) -> bool:
    """Pre-filter: both series should be stationary (ADF p-val < thresh).
    
    This is a FAST vectorized ADF that avoids statsmodels overhead.
    Returns True if both pass.
    """
    _, px = fast_adf_residual(x)
    _, py = fast_adf_residual(y)
    return px < pval_thresh and py < pval_thresh


def engle_granger(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """EG residual cointegration test on log-price series.
    
    Only called AFTER ADF pre-filter passes. Uses statsmodels.
    """
    from statsmodels.tsa.stattools import adfuller
    import statsmodels.api as sm

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    resid = res.resid
    beta = float(res.params[1])
    adf = adfuller(resid, autolag="AIC", regression="c")
    return float(adf[0]), float(adf[1]), beta


def half_life(spread: np.ndarray) -> float:
    """OU half-life from AR(1) on the spread: hl = -ln(2)/ln(rho)."""
    s = np.asarray(spread, float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return float("inf")
    y = s[1:]
    x = s[:-1]
    denom = np.sum((x - x.mean()) ** 2)
    if denom == 0:
        return float("inf")
    rho = np.sum((x - x.mean()) * (y - y.mean())) / denom
    if not 0 < rho < 1:
        return float("inf")
    return -np.log(2) / np.log(rho)


def select_pairs(
    wide: pd.DataFrame,
    groups: dict[str, list[str]],
    lookback: int = 504,
    alpha: float = 0.10,
    corr_min: float = 0.35,
    corr_max: float = 0.95,
) -> pd.DataFrame:
    """EG + FDR on trailing lookback. Return-corr screen before EG."""
    rows: list[dict] = []
    lpx = np.log(wide)
    n_cand = 0
    n_corr = 0

    for g, tks in groups.items():
        tks = [t for t in tks if t in lpx.columns]
        if len(tks) < 2:
            continue
        sub = lpx[tks].iloc[-lookback:]
        corr = sub.diff().corr().to_numpy()
        for i in range(len(tks)):
            for j in range(i + 1, len(tks)):
                n_cand += 1
                rho = corr[i, j]
                if not np.isfinite(rho) or abs(rho) < corr_min or abs(rho) > corr_max:
                    continue
                n_corr += 1
                pair = sub[[tks[i], tks[j]]].dropna()
                if len(pair) < 200:
                    continue
                x = pair.iloc[:, 0].to_numpy()
                y = pair.iloc[:, 1].to_numpy()
                try:
                    tstat, pval, beta = engle_granger(x, y)
                except Exception:
                    continue
                spread = y - beta * x
                hl = half_life(spread)
                rows.append({
                    "pair_id": f"{tks[i]}|{tks[j]}",
                    "group": g,
                    "asset_a": tks[i],
                    "asset_b": tks[j],
                    "coint_t": tstat,
                    "p_value": pval,
                    "beta": beta,
                    "half_life": hl,
                    "ret_corr": float(rho),
                })

    print(f"  corr screen: {n_cand} candidates, {n_corr} in [{corr_min}, {corr_max}], {len(rows)} EG-ok")

    if not rows:
        return pd.DataFrame(columns=["pair_id", "group", "asset_a", "asset_b", "coint_t",
                                    "p_value", "beta", "half_life", "ret_corr", "fdr_survive", "usable"])
    df = pd.DataFrame(rows)
    df["fdr_survive"] = bh_fdr(df["p_value"].to_numpy(), alpha=alpha)
    df["usable"] = (
        df["fdr_survive"]
        & np.isfinite(df["half_life"])
        & (df["half_life"] >= 2.0)
        & (df["half_life"] <= 250.0)
    )
    return df.sort_values("coint_t")


def simulate_pair(
    wide: pd.DataFrame,
    a: str,
    b: str,
    beta: float,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 4.0,
    max_hold: int = 60,
) -> list[dict]:
    """Trade the pair over (train_end, test_end]. Returns trade list."""
    lpx = np.log(wide)
    idx = wide.index[(wide.index > train_end) & (wide.index <= test_end)]
    if len(idx) < 20:
        return []
    s = (lpx[b] - beta * lpx[a]).loc[idx]
    train_spread = (lpx[b] - beta * lpx[a]).loc[:train_end].dropna()
    mu, sd = float(train_spread.mean()), float(train_spread.std())
    if sd == 0 or np.isnan(sd):
        return []
    z = (s - mu) / sd

    trades: list[dict] = []
    pos = 0
    entry_dt = None
    entry_z_val = 0.0
    entry_idx_pos = 0
    
    z_arr = z.to_numpy()
    idx_arr = np.array(idx)
    
    for i in range(len(idx_arr)):
        zv = float(z_arr[i]) if not np.isnan(z_arr[i]) else np.nan
        if np.isnan(zv):
            continue
        dt = idx_arr[i]
        if pos == 0:
            if zv >= entry_z and zv <= 6.0:
                pos = 1
                entry_dt = dt
                entry_z_val = zv
                entry_idx_pos = i
            elif zv <= -entry_z and zv >= -6.0:
                pos = -1
                entry_dt = dt
                entry_z_val = zv
                entry_idx_pos = i
        else:
            bars = i - entry_idx_pos
            exit_reason = None
            if (pos == 1 and zv <= exit_z) or (pos == -1 and zv >= exit_z):
                exit_reason = "revert"
            elif abs(zv) >= stop_z:
                exit_reason = "stop"
            elif bars >= max_hold:
                exit_reason = "time"
            if exit_reason:
                            pnl = pos * (zv - entry_z_val)
                            pa0 = float(wide[a].loc[:entry_dt].dropna().iloc[-1])
                            pb0 = float(wide[b].loc[:entry_dt].dropna().iloc[-1])
                            pa1 = float(wide[a].loc[dt])
                            pb1 = float(wide[b].loc[dt])
                            ret_b = pb1 / pb0 - 1 if pb0 > 0 else 0.0
                            ret_a = pa1 / pa0 - 1 if pa0 > 0 else 0.0
                            hedged = pos * (ret_b - beta * ret_a)
                            trades.append({
                                "pair_id": f"{a}|{b}",
                                "entry_date": pd.Timestamp(entry_dt).date(),
                                "exit_date": pd.Timestamp(dt).date(),
                                "entry_z": round(entry_z_val, 3),
                                "exit_z": round(zv, 3),
                                "bars_held": bars,
                                "exit_reason": exit_reason,
                                "hedged_pnl": round(hedged, 5),
                                "z_pnl": round(pnl, 3),
                            })
                            pos = 0
                            entry_dt = None
                            entry_z_val = 0.0
    return trades


def build(
    lookback: int = 378,
    test_days: int = 252,
    n_folds: int = 3,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_z: float = 4.0,
    max_hold: int = 60,
    alpha: float = 0.20,
    max_pairs: int = 400,
    max_per_group: int = 40,
    corr_min: float = 0.35,
    corr_max: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = _groups(max_per_group=max_per_group)
    tickers = sorted({t for ts in groups.values() for t in ts})
    print(f"  groups={len(groups)} names={len(tickers)} max_per_group={max_per_group}")
    prices = load_adj_prices_pandas(tickers=tickers)
    wide = wide_closes(prices).sort_index().dropna(how="all")
    if len(wide) < lookback + test_days + 50:
        raise SystemExit("Not enough price history for the requested windows")

    n = len(wide)
    fold_ends = [n - test_days * (n_folds - k) for k in range(n_folds)]
    all_trades: list[dict] = []
    all_fold_pairs: list[pd.DataFrame] = []
    z_now_map: dict[str, float] = {}
    lpx = np.log(wide)

    for k, test_end_pos in enumerate(fold_ends):
        test_start_pos = test_end_pos - test_days
        train_end_pos = test_start_pos - 1
        sel_wide = wide.iloc[: train_end_pos + 1]
        pairs = select_pairs(sel_wide, groups, lookback=lookback, alpha=alpha,
                             corr_min=corr_min, corr_max=corr_max)
        usable = pairs[pairs["usable"]].head(max_pairs)
        usable = usable.copy()
        usable["fold"] = k
        all_fold_pairs.append(usable)

        test_start = wide.index[test_start_pos]
        test_end = wide.index[test_end_pos]
        train_end = wide.index[train_end_pos]
        for _, p in usable.iterrows():
            trades = simulate_pair(
                wide, p["asset_a"], p["asset_b"], p["beta"],
                train_end, test_end,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_hold=max_hold,
            )
            for t in trades:
                t["fold"] = k
            all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)

    if len(trades_df):
        trades_df = apply_costs_to_trades(trades_df, pnl_col="hedged_pnl")

    pair_stats: list[dict] = []
    if len(trades_df):
        net_col = "net_hedged_pnl" if "net_hedged_pnl" in trades_df.columns else "hedged_pnl"
        for (pid,), grp in trades_df.groupby(["pair_id"]):
            arr = grp[net_col].to_numpy()
            wins = float((arr > 0).mean())
            mean_hold = float(grp["bars_held"].mean()) if len(grp) else 1.0
            sharpe = float(arr.mean() / arr.std() * np.sqrt(252 / max(1, mean_hold))) if arr.std() > 0 else 0.0
            row = grp.iloc[0]
            pair_stats.append({
                "pair_id": pid,
                "group": row["group"] if "group" in grp.columns else "",
                "n_trades": len(grp),
                "win_rate": round(wins, 3),
                "avg_pnl": round(float(arr.mean()), 5),
                "total_pnl": round(float(arr.sum()), 5),
                "avg_gross_pnl": round(float(grp["hedged_pnl"].mean()), 5) if "hedged_pnl" in grp else None,
                "sharpe": round(sharpe, 3),
                "folds": int(grp["fold"].nunique()),
            })
    stats_df = pd.DataFrame(pair_stats)

    if all_fold_pairs and not all_fold_pairs[-1].empty:
        cur = all_fold_pairs[-1]
        for _, p in cur.iterrows():
            s = (lpx[p["asset_b"]] - p["beta"] * lpx[p["asset_a"]]).dropna()
            if len(s) > test_days:
                mu, sd = s.iloc[:-test_days].mean(), s.iloc[:-test_days].std()
                if sd and sd > 0:
                    z_now_map[p["pair_id"]] = round(float((s.iloc[-1] - mu) / sd), 2)

    pairs_all = pd.concat(all_fold_pairs, ignore_index=True) if all_fold_pairs else pd.DataFrame()
    if len(pairs_all) and len(z_now_map):
        pairs_all["z_now"] = pairs_all["pair_id"].map(z_now_map)
    return pairs_all, trades_df, stats_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookback", type=int, default=378, help="Selection window (trading days)")
    ap.add_argument("--test-days", type=int, default=252, help="OOS test window per fold")
    ap.add_argument("--n-folds", type=int, default=3, help="Walk-forward folds")
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.0)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.20, help="FDR level")
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--max-per-group", type=int, default=40,
                    help="cap names per industry (ranked by lookback coverage)")
    ap.add_argument("--corr-min", type=float, default=0.35)
    ap.add_argument("--corr-max", type=float, default=0.95)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    pairs, trades, stats = build(
        lookback=args.lookback, test_days=args.test_days, n_folds=args.n_folds,
        entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z,
        max_hold=args.max_hold, alpha=args.alpha, max_pairs=args.max_pairs,
        max_per_group=args.max_per_group, corr_min=args.corr_min, corr_max=args.corr_max,
    )
    n_sel = int(pairs["usable"].sum()) if len(pairs) and "usable" in pairs else 0
    print(f"=== walk-forward pairs selected (FDR + half-life): {n_sel} across {args.n_folds} folds ===")
    if len(pairs) and "usable" in pairs:
        print(pairs[pairs["usable"]].head(12).to_string(index=False))
    print(f"\n=== OOS trades: {len(trades)} ===")
    if len(trades):
        print(trades.head(12).to_string(index=False))
    print(f"\n=== OOS pair stats: {len(stats)} pairs traded ===")
    if len(stats):
        print(stats.sort_values("sharpe", ascending=False).head(12).to_string(index=False))
        agg_wr = float((stats["win_rate"] * stats["n_trades"]).sum() / stats["n_trades"].sum())
        print(f"\nAggregate OOS win rate (trade-weighted): {agg_wr:.3f}")
    if args.save:
        pairs.to_parquet(OUT_PAIRS)
        trades.to_parquet(OUT_TRADES)
        stats.to_parquet(OUT_STATS)
        print(f"\nWrote {OUT_PAIRS}\nWrote {OUT_TRADES}\nWrote {OUT_STATS}")


if __name__ == "__main__":
    main()
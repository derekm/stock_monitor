#!/usr/bin/env python3
"""
pair_engine.py — Pair / relative-value engine: cointegration + residual
mean-reversion inside industry groups, with stops and time exits.

  1. Universe: pairs inside the same industry, cap --max-per-group (coverage-ranked).
  2. Return-correlation screen before Engle-Granger.
  3. Batched OLS + fixed-lag residual ADF (no statsmodels AIC).
  4. FDR across surviving pairs; walk-forward OOS trading.

Daily DAG: --max-per-group 40 --n-folds 1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns
from cv_utils import bh_fdr
from cost_model import apply_costs_to_trades
from statsmodels.tsa.stattools import mackinnonp

PRICES = DATA_DIR / "daily_prices/"
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
    """Fixed-lag ADF on a residual (constant, 1 lag of dy). MacKinnon p-value."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return 0.0, 1.0
    dy = np.diff(x)
    x_lag = x[:-1]
    dy_lag = np.empty_like(dy)
    dy_lag[0] = 0.0
    dy_lag[1:] = dy[:-1]
    X = np.column_stack([np.ones(n - 1), x_lag, dy_lag])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        resid = dy - X @ beta
        dof = n - 3
        sse = float(np.dot(resid, resid))
        sigma2 = sse / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
        t_stat = float(beta[1] / se) if se > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    return t_stat, float(mackinnonp(t_stat, regression="c", N=1))


def engle_granger(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """EG: OLS y~x then fixed-lag ADF on residual. No statsmodels."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        return 0.0, 1.0, np.nan
    xd, yd = x - x.mean(), y - y.mean()
    varx = float(np.dot(xd, xd))
    if varx <= 1e-18:
        return 0.0, 1.0, np.nan
    beta = float(np.dot(xd, yd) / varx)
    resid = y - (y.mean() - beta * x.mean()) - beta * x
    tstat, pval = fast_adf_residual(resid)
    return tstat, pval, beta


def _eg_batch(arr: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
              chunk: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise-complete OLS + ADF. Chunks fat groups so resid stays in RAM."""
    i_idx = np.asarray(i_idx)
    j_idx = np.asarray(j_idx)
    n = int(i_idx.size)
    if n > chunk:
        ts, pv, be = [], [], []
        for s in range(0, n, chunk):
            t, p, b = _eg_batch(arr, i_idx[s:s + chunk], j_idx[s:s + chunk], chunk=chunk)
            ts.append(t); pv.append(p); be.append(b)
        return np.concatenate(ts), np.concatenate(pv), np.concatenate(be)
    xi = arr[:, i_idx]
    xj = arr[:, j_idx]
    ok = np.isfinite(xi) & np.isfinite(xj)
    cnt = ok.sum(axis=0)
    xi_m = np.where(ok, xi, np.nan)
    xj_m = np.where(ok, xj, np.nan)
    mx = np.nanmean(xi_m, axis=0)
    my = np.nanmean(xj_m, axis=0)
    xd = np.where(ok, xi - mx, 0.0)
    yd = np.where(ok, xj - my, 0.0)
    varx = np.sum(xd * xd, axis=0)
    cov = np.sum(xd * yd, axis=0)
    beta = np.where(varx > 1e-18, cov / varx, np.nan)
    alpha = my - beta * mx
    resid = np.where(ok, xj - alpha - beta * xi, np.nan)
    n_p = resid.shape[1]
    tstat = np.zeros(n_p)
    pval = np.ones(n_p)
    for p in range(n_p):
        if cnt[p] < 200 or not np.isfinite(beta[p]):
            tstat[p], pval[p] = 0.0, 1.0
            continue
        tstat[p], pval[p] = fast_adf_residual(resid[:, p])
    return tstat, pval, beta


def half_life(spread: np.ndarray) -> float:
    """OU half-life from AR(1) on the spread: hl = -ln(2)/ln(rho)."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
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


def _select_one_group(g, tks, arr, cols, lookback, corr_min, corr_max):
    idx = [cols[t] for t in tks if t in cols]
    names = [t for t in tks if t in cols]
    if len(idx) < 2:
        return [], 0, 0
    sub = arr[-lookback:, idx]
    d = np.diff(sub, axis=0)
    # pairwise corr of diffs, nan-safe
    d0 = d - np.nanmean(d, axis=0)
    d0 = np.where(np.isfinite(d0), d0, 0.0)
    valid = np.isfinite(d).astype(float)
    nobs = valid.T @ valid
    gram = d0.T @ d0
    denom = np.sqrt(np.clip(np.diag(gram), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = gram / np.outer(denom, denom)
    corr = np.where(nobs >= 50, corr, np.nan)
    ii, jj = np.triu_indices(len(names), k=1)
    rho = corr[ii, jj]
    keep = np.isfinite(rho) & (np.abs(rho) >= corr_min) & (np.abs(rho) <= corr_max)
    n_cand = int(len(ii))
    n_corr = int(keep.sum())
    if n_corr == 0:
        return [], n_cand, 0
    ii, jj, rho = ii[keep], jj[keep], rho[keep]
    tstat, pval, beta = _eg_batch(sub, ii, jj)
    rows = []
    for k in range(len(ii)):
        if not np.isfinite(beta[k]) or pval[k] >= 1.0:
            continue
        a, b = names[ii[k]], names[jj[k]]
        xa, xb = sub[:, ii[k]], sub[:, jj[k]]
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 200:
            continue
        spread = xb[ok] - beta[k] * xa[ok]
        hl = half_life(spread)
        rows.append({
            "pair_id": f"{a}|{b}",
            "group": g,
            "asset_a": a,
            "asset_b": b,
            "coint_t": float(tstat[k]),
            "p_value": float(pval[k]),
            "beta": float(beta[k]),
            "half_life": hl,
            "ret_corr": float(rho[k]),
        })
    return rows, n_cand, n_corr


def select_pairs(
    wide: pd.DataFrame,
    groups: dict[str, list[str]],
    lookback: int = 504,
    alpha: float = 0.10,
    corr_min: float = 0.35,
    corr_max: float = 0.95,
) -> pd.DataFrame:
    """EG + FDR on trailing lookback. Return-corr screen before EG."""
    lpx = np.log(wide.to_numpy(dtype=float))
    cols = {c: i for i, c in enumerate(wide.columns)}
    items = [(g, tks) for g, tks in groups.items() if len(tks) >= 2]
    rows: list[dict] = []
    n_cand = n_corr = 0
    n_g = len(items)
    workers = min(10, max(1, n_g))
    print(f"  select_pairs groups={n_g} workers={workers} lookback={lookback}")

    def _job(item):
        g, tks = item
        return _select_one_group(g, tks, lpx, cols, lookback, corr_min, corr_max)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, it) for it in items]
        for fut in as_completed(futs):
            r, c, cc = fut.result()
            rows.extend(r)
            n_cand += c
            n_corr += cc
            done += 1
            if done % 20 == 0 or done == n_g:
                print(f"  groups {done}/{n_g}  corr-pass {n_corr}  eg-ok {len(rows)}")

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
    print(f"  price panel {wide.shape[0]} dates x {wide.shape[1]} names")
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
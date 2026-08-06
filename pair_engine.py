#!/usr/bin/env python3
"""
pair_engine.py — Pair / relative-value engine: cointegration + residual
mean-reversion inside industry groups, with stops and time exits.

Pipeline (all OOS by construction):
  1. Universe: pairs inside same industry group (min 2 names per group).
  2. Selection on TRAILING window only: Engle-Granger cointegration t-stat
     (statsmodels) on log prices; half-life of the OU spread; FDR-corrected
     across all candidate pairs (Benjamini-Hochberg, alpha=0.10).
  3. Trade the NEXT window (walk-forward): z-score entry (+/-2), exit on
     reversion to 0, stop at z +/-4, time exit after max_hold bars.
  4. Report OOS stats per pair + aggregate; never in-sample.

Outputs:
  pair_engine_pairs.csv     pair_id, group, coint_t, p-value, half_life,
                            beta, z_now, fdr_survive
  pair_engine_trades.csv    per executed trade: entry/exit dates, z in/out,
                            pnl (hedged), bars_held, exit_reason
  pair_engine_stats.csv     per-pair OOS: n_trades, win_rate, avg_pnl,
                            total_pnl, sharpe, last_z
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
OUT_PAIRS = DATA_DIR / "pair_engine_pairs.csv"
OUT_TRADES = DATA_DIR / "pair_engine_trades.csv"
OUT_STATS = DATA_DIR / "pair_engine_stats.csv"


def _groups() -> dict[str, list[str]]:
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    if stocks.empty or "ticker" not in stocks.columns:
        return {}
    groups: dict[str, list[str]] = {}
    for _, r in stocks.iterrows():
        g = str(r.get("industry") or r.get("sector") or "unknown")
        tk = str(r["ticker"]).upper()
        groups.setdefault(g, []).append(tk)
    return {g: sorted(set(ts)) for g, ts in groups.items() if len(set(ts)) >= 2}


def engle_granger(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """EG residual cointegration test on log-price series.

    Regress y on x, ADF-test the residual (statsmodels). Returns
    (t-stat, p-value). Lower t = more cointegrated.
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


def select_pairs(wide: pd.DataFrame, groups: dict[str, list[str]], lookback: int = 504, alpha: float = 0.10) -> pd.DataFrame:
    """EG + FDR on trailing lookback window. Returns pair candidates."""
    rows: list[dict] = []
    lpx = np.log(wide)
    for g, tks in groups.items():
        tks = [t for t in tks if t in lpx.columns]
        for i in range(len(tks)):
            for j in range(i + 1, len(tks)):
                a, b = tks[i], tks[j]
                sa = lpx[a].dropna().tail(lookback)
                sb = lpx[b].dropna().tail(lookback)
                # align on common index
                idx = sa.index.intersection(sb.index)
                if len(idx) < 200:
                    continue
                x = sa.loc[idx].to_numpy()
                y = sb.loc[idx].to_numpy()
                try:
                    tstat, pval, beta = engle_granger(x, y)
                except Exception:  # noqa: BLE001 - statsmodels edge cases
                    continue
                spread = y - beta * x
                hl = half_life(spread)
                rows.append({
                    "pair_id": f"{a}|{b}",
                    "group": g,
                    "asset_a": a,
                    "asset_b": b,
                    "coint_t": tstat,
                    "p_value": pval,
                    "beta": beta,
                    "half_life": hl,
                })
    if not rows:
        return pd.DataFrame(columns=["pair_id", "group", "asset_a", "asset_b", "coint_t", "p_value", "beta", "half_life", "fdr_survive"])
    df = pd.DataFrame(rows)
    # FDR across ALL tested pairs
    pvals = df["p_value"].to_numpy()
    df["fdr_survive"] = bh_fdr(pvals, alpha=alpha)
    # usable = survives FDR AND has a tradeable mean-reversion horizon.
    # half_life < 3d = white noise (no persistence to harvest);
    # half_life > 200d = relationship too slow to trade within a year.
    # These bounds are the anti-spurious filter: EG p-values are ~0 for
    # nearly every pair (trending series fit well), so FDR alone lets
    # nonsense pairs through (e.g. startup vs blue-chip with hl~0.1d).
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
    # spread from beta estimated in-sample (fixed)
    s = (lpx[b] - beta * lpx[a]).loc[idx]
    # z-score using in-sample spread stats
    train_spread = (lpx[b] - beta * lpx[a]).loc[:train_end].dropna()
    mu, sd = float(train_spread.mean()), float(train_spread.std())
    if sd == 0 or np.isnan(sd):
        return []
    z = (s - mu) / sd

    trades: list[dict] = []
    pos = 0  # +1 long spread (long b, short a), -1 short spread
    entry_dt = None
    entry_z_val = 0.0
    for i, dt in enumerate(idx):
        zv = float(z.loc[dt]) if dt in z.index else np.nan
        if np.isnan(zv):
            continue
        if pos == 0:
            # |z| beyond 6 = relationship broke (not reversion); skip, don't trade
            if zv >= entry_z and zv <= 6.0:
                pos = 1
                entry_dt = dt
                entry_z_val = zv
            elif zv <= -entry_z and zv >= -6.0:
                pos = -1
                entry_dt = dt
                entry_z_val = zv
        else:
            bars = i - list(idx).index(entry_dt) if entry_dt in idx else 0
            exit_reason = None
            if (pos == 1 and zv <= exit_z) or (pos == -1 and zv >= exit_z):
                exit_reason = "revert"
            elif abs(zv) >= stop_z:
                exit_reason = "stop"
            elif bars >= max_hold:
                exit_reason = "time"
            if exit_reason:
                pnl = pos * (zv - entry_z_val)  # normalized z-pnl proxy
                # hedged dollar pnl: pos * (ret_b - beta*ret_a) over hold
                pa0 = float(wide[a].loc[:entry_dt].dropna().iloc[-1])
                pb0 = float(wide[b].loc[:entry_dt].dropna().iloc[-1])
                pa1 = float(wide[a].loc[dt])
                pb1 = float(wide[b].loc[dt])
                ret_b = pb1 / pb0 - 1 if pb0 > 0 else 0.0
                ret_a = pa1 / pa0 - 1 if pa0 > 0 else 0.0
                hedged = pos * (ret_b - beta * ret_a)
                trades.append({
                    "pair_id": f"{a}|{b}",
                    "entry_date": entry_dt.date(),
                    "exit_date": dt.date(),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = _groups()
    tickers = sorted({t for ts in groups.values() for t in ts})
    prices = load_adj_prices_pandas(tickers=tickers)
    wide = wide_closes(prices).sort_index().dropna(how="all")
    if len(wide) < lookback + test_days + 50:
        raise SystemExit("Not enough price history for the requested windows")

    # Walk-forward: n_folds non-overlapping OOS windows. Each fold selects
    # pairs on the trailing lookback ending AT the fold boundary, then trades
    # the following test_days. Selection and trading are always disjoint.
    n = len(wide)
    fold_ends = [n - test_days * (n_folds - k) for k in range(n_folds)]  # ascending
    all_trades: list[dict] = []
    all_fold_pairs: list[pd.DataFrame] = []
    z_now_map: dict[str, float] = {}
    lpx = np.log(wide)

    for k, test_end_pos in enumerate(fold_ends):
        test_start_pos = test_end_pos - test_days  # inclusive test start
        train_end_pos = test_start_pos - 1          # selection ends day before test
        # selection: trailing lookback ending at train_end_pos
        sel_wide = wide.iloc[: train_end_pos + 1]
        pairs = select_pairs(sel_wide, groups, lookback=lookback, alpha=alpha)
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

    # net of costs: 10bps round trip per pair trade + borrow on short leg
    if len(trades_df):
        trades_df = apply_costs_to_trades(trades_df, pnl_col="hedged_pnl")

    # pair stats aggregated across folds (on NET pnl)
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

    # live z for the latest selected pairs (last fold's selection, current spread)
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
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    pairs, trades, stats = build(
        lookback=args.lookback, test_days=args.test_days, n_folds=args.n_folds,
        entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z,
        max_hold=args.max_hold, alpha=args.alpha, max_pairs=args.max_pairs,
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
        pairs.to_csv(OUT_PAIRS, index=False)
        trades.to_csv(OUT_TRADES, index=False)
        stats.to_csv(OUT_STATS, index=False)
        print(f"\nWrote {OUT_PAIRS}\nWrote {OUT_TRADES}\nWrote {OUT_STATS}")


if __name__ == "__main__":
    main()

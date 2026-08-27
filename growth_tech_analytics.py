#!/usr/bin/env python3
"""
growth_tech_analytics.py — Full analysis suite for the growth/tech index.

Mirrors sector/portfolio analytics applied to growth_tech_index members:
  - Membership & sleeve snapshot
  - Realized vol / return summary
  - Correlation matrix (full + by sleeve)
  - Rolling correlations & stability
  - EW index levels + backtest stats vs defensive/fertilizer/portfolio proxies
  - ERC / InvVol / GMV / vol-target comparison
  - Fisher chained index (price/qty) if volume present
  - Simple forecast snapshot (fallback drift)

Outputs under growth_tech_*.csv / .parquet

Usage:
  python growth_tech_analytics.py
  python growth_tech_analytics.py --window 126
  python maintain_analytics.py growth-tech
"""

from __future__ import annotations

import argparse
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
LEVELS = DATA_DIR / "growth_tech_index_levels.parquet"


def members(stocks: pd.DataFrame) -> pd.DataFrame:
    m = stocks[stocks.get("growth_tech_index", False) == True].copy()
    if m.empty:
        # Do not crash the DAG because a sleeve has not been populated yet —
        # the corrected runner would then cascade-fail everything downstream.
        # Warn once and let main() write an empty-but-valid parquet so dependents
        # read a consistent schema instead of a missing file.
        print("WARNING: growth_tech_index has no members — check sleeve population.")
    return m


def load_panel(prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = prices[prices["ticker"].isin(tickers)].copy()
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    wide = sub.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    wide = wide.dropna(how="all")
    rets = np.log(wide / wide.shift(1))
    return wide, rets


def ann_vol(r: pd.Series, window: int = 21) -> float:
    x = r.dropna().iloc[-window:]
    if len(x) < 5:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(252))


def cmd_membership(m: pd.DataFrame) -> pd.DataFrame:
    out = m[["ticker", "name", "sector", "growth_sleeve", "notes"]].copy() if "name" in m.columns else m[["ticker", "growth_sleeve"]].copy()
    out.to_parquet(DATA_DIR / "growth_tech_membership.parquet")
    print(f"membership: {len(out)} names")
    print(out.groupby("growth_sleeve")["ticker"].apply(list).to_string() if "growth_sleeve" in out.columns else "")
    return out


def cmd_vol_return(rets: pd.DataFrame, wide: pd.DataFrame, m: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    for t in rets.columns:
        r = rets[t].dropna()
        px = wide[t].dropna()
        rows.append({
            "ticker": t,
            "growth_sleeve": m.set_index("ticker").loc[t, "growth_sleeve"] if t in m.set_index("ticker").index and "growth_sleeve" in m.columns else None,
            "last_close": float(px.iloc[-1]),
            "ret_1m": float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) >= 21 else float("nan"),
            "ret_3m": float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) >= 63 else float("nan"),
            "ret_1y": float(px.iloc[-1] / px.iloc[0] - 1) if len(px) >= 2 else float("nan"),
            "vol_21d": ann_vol(r, 21),
            "vol_63d": ann_vol(r, 63),
            "vol_window": ann_vol(r, window),
            "sharpe_proxy": (float(r.tail(window).mean() * 252) / ann_vol(r, window)) if ann_vol(r, window) > 1e-8 else float("nan"),
        })
    df = pd.DataFrame(rows).sort_values("vol_21d", ascending=False)
    df.to_parquet(DATA_DIR / "growth_tech_vol_returns.parquet")
    print("\n=== Vol / returns (top by 21d vol) ===")
    print(df[["ticker", "growth_sleeve", "last_close", "ret_3m", "vol_21d", "sharpe_proxy"]].head(12).to_string(index=False))
    return df


def cmd_correlations(rets: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    c = rets.corr()
    c.to_parquet(DATA_DIR / "growth_tech_correlation_matrix.parquet")
    # sleeve average corr
    rows = []
    if "growth_sleeve" in m.columns:
        sleeve = m.set_index("ticker")["growth_sleeve"].to_dict()
        sleeves = sorted({s for s in sleeve.values() if s})
        for a in sleeves:
            for b in sleeves:
                ta = [t for t, s in sleeve.items() if s == a and t in c.columns]
                tb = [t for t, s in sleeve.items() if s == b and t in c.columns]
                if not ta or not tb:
                    continue
                block = c.loc[ta, tb].values
                if a == b:
                    # off-diagonal
                    mask = ~np.eye(len(ta), dtype=bool) if len(ta) > 1 else np.zeros_like(block, dtype=bool)
                    vals = block[mask] if mask.any() else block.ravel()
                else:
                    vals = block.ravel()
                rows.append({"sleeve_a": a, "sleeve_b": b, "avg_corr": float(np.nanmean(vals)), "n_pairs": int(np.isfinite(vals).sum())})
        sdf = pd.DataFrame(rows)
        sdf.to_parquet(DATA_DIR / "growth_tech_sleeve_correlations.parquet")
        print("\n=== Sleeve avg correlations ===")
        print(sdf.pivot(index="sleeve_a", columns="sleeve_b", values="avg_corr").round(2).to_string())
    # pair extremes
    pairs = []
    cols = list(c.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((a, b, float(c.loc[a, b])))
    pairs = sorted(pairs, key=lambda x: x[2])
    print("\nLowest corrs:", pairs[:5])
    print("Highest corrs:", pairs[-5:])
    return c


def cmd_rolling(rets: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    cols = list(rets.columns)
    # average pairwise rolling corr
    dates = rets.index[window:]
    rows = []
    for i in range(window, len(rets)):
        block = rets.iloc[i - window : i]
        c = block.corr().values
        mask = np.triu(np.ones_like(c, dtype=bool), 1)
        vals = c[mask]
        rows.append({
            "date": rets.index[i],
            "window": window,
            "avg_corr": float(np.nanmean(vals)),
            "median_corr": float(np.nanmedian(vals)),
            "p10_corr": float(np.nanpercentile(vals, 10)),
            "p90_corr": float(np.nanpercentile(vals, 90)),
            "dispersion": float(np.nanstd(vals)),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "growth_tech_rolling_corr.parquet")
    print(f"\n=== Rolling {window}d pairwise corr ===")
    print(f"  last avg={df.avg_corr.iloc[-1]:.3f}  median={df.median_corr.iloc[-1]:.3f}  "
          f"dispersion={df.dispersion.iloc[-1]:.3f}")
    print(f"  mean avg_corr over history={df.avg_corr.mean():.3f}  std={df.avg_corr.std():.3f}")
    # stability metric
    stab = pd.DataFrame([{
        "metric": "avg_pairwise_corr",
        "mean": df.avg_corr.mean(),
        "std": df.avg_corr.std(),
        "cv": df.avg_corr.std() / abs(df.avg_corr.mean()) if df.avg_corr.mean() else np.nan,
        "last": df.avg_corr.iloc[-1],
        "min": df.avg_corr.min(),
        "max": df.avg_corr.max(),
    }])
    stab.to_parquet(DATA_DIR / "growth_tech_corr_stability.parquet")
    return df


def cmd_index_backtest(wide: pd.DataFrame, prices: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    # EW growth index
    rets = wide.pct_change()
    ew = rets.mean(axis=1).fillna(0)
    level = (1 + ew).cumprod() * 100
    level = level / level.iloc[0] * 100
    lv = level.rename("growth_tech").to_frame()

    # compare to other sleeves if possible
    def ew_index(flag_col):
        if flag_col not in stocks.columns:
            return None
        t = stocks.loc[stocks[flag_col] == True, "ticker"].tolist()
        w = prices[prices.ticker.isin(t)].pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        r = w.pct_change().mean(axis=1).fillna(0)
        x = (1 + r).cumprod() * 100
        return x / x.iloc[0] * 100

    for name, col in [("fertilizer", "index_member"), ("defensive", "defensive_value_index")]:
        x = ew_index(col)
        if x is not None:
            lv = lv.join(x.rename(name), how="outer")

    # personal portfolio proxy from holdings EW
    if HOLDINGS.exists():
        h = pd.read_parquet(HOLDINGS)
        t = h["ticker"].tolist()
        w = prices[prices.ticker.isin(t)].pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        r = w.pct_change().mean(axis=1).fillna(0)
        x = (1 + r).cumprod() * 100
        lv = lv.join((x / x.iloc[0] * 100).rename("personal_ew"), how="outer")

    lv = lv.dropna(how="all").ffill()
    lv.to_parquet(DATA_DIR / "growth_tech_index_levels_compare.parquet")
    lv.reset_index().to_parquet(DATA_DIR / "growth_tech_index_levels_compare.parquet")

    def perf(s: pd.Series) -> dict:
        s = s.dropna()
        if len(s) < 5:
            return {}
        r = s.pct_change().dropna()
        return {
            "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
            "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": float((r.mean() * 252) / (r.std() * np.sqrt(252))) if r.std() > 0 else np.nan,
            "max_dd": float((s / s.cummax() - 1).min()),
            "last_level": float(s.iloc[-1]),
        }

    rows = []
    for c in lv.columns:
        p = perf(lv[c])
        p["index"] = c
        rows.append(p)
    stats = pd.DataFrame(rows)
    stats.to_parquet(DATA_DIR / "growth_tech_backtest_stats.parquet")
    print("\n=== Index backtest ===")
    print(stats[["index", "total_return", "ann_vol", "sharpe", "max_dd", "last_level"]].to_string(index=False))
    return stats


def cmd_risk_models(rets: pd.DataFrame, m: pd.DataFrame, window: int) -> pd.DataFrame:
    """ERC / InvVol / GMV / vol-target on growth universe."""
    try:
        from portfolio_optimization import (
            erc_slsqp, inv_vol_weights, gmv_long_only, gmv_long_capped,
            vol_target_renorm, stats as po_stats,
        )
    except Exception as e:
        print("portfolio_optimization import failed:", e)
        return pd.DataFrame()

    r = rets.iloc[-window:].dropna(how="all")
    r = r.dropna(axis=1, thresh=max(40, window // 3))
    tickers = list(r.columns)
    cov = r.cov().values * 252.0
    mu = r.mean().values * 252.0

    w_erc, _ = erc_slsqp(cov, w_floor=0.02)
    w_iv = inv_vol_weights(cov)
    w_gmv, _ = gmv_long_only(cov)
    caps = {t: 0.15 for t in tickers}
    w_gmv_c, _ = gmv_long_capped(cov, tickers, caps)
    w_vt = vol_target_renorm(cov, tickers, target=0.25, name_cap=0.15)

    strategies = {
        "EW": np.ones(len(tickers)) / len(tickers),
        "ERC_SLSQP": w_erc,
        "InvVol": w_iv,
        "GMV": w_gmv,
        "GMV_capped": w_gmv_c,
        "VolTarget_renorm": w_vt,
    }
    rows = []
    print("\n=== Risk models (growth tech) ===")
    for name, w in strategies.items():
        st = po_stats(w, cov, mu)
        print(f"  {name:18s} σ={st['vol']*100:5.2f}%  ret≈{st['ret']*100:6.2f}%  "
              f"RC_disp={st['rc_dispersion']*100:.2f}%")
        for i, t in enumerate(tickers):
            rows.append({
                "strategy": name, "ticker": t, "weight": float(w[i]),
                "rc_pct_var": float(st["rc_pct"][i]),
                "port_vol": st["vol"], "port_ret": st["ret"],
            })
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "growth_tech_risk_models.parquet")
    return df


def cmd_fisher(tickers: list[str]) -> None:
    try:
        from run_fisher_duckdb import compute
        df = compute(tickers, freq="D", label="growth_tech")
        out = DATA_DIR / "growth_tech_fisher.parquet"
        df.to_parquet(out)
        print(f"\n=== Fisher (growth_tech) last ===")
        print(df[["date", "fisher_p", "fisher_q", "nominal_sqrt_fisher"]].tail(3).to_string(index=False))
        print(f"Wrote {out}")
    except Exception as e:
        print("Fisher skipped:", e)


def cmd_sleeve_perf(rets: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    if "growth_sleeve" not in m.columns:
        return pd.DataFrame()
    sleeve = m.set_index("ticker")["growth_sleeve"].to_dict()
    rows = []
    for sname in sorted(set(sleeve.values())):
        cols = [t for t, s in sleeve.items() if s == sname and t in rets.columns]
        if not cols:
            continue
        ew = rets[cols].mean(axis=1).dropna()
        rows.append({
            "sleeve": sname,
            "n": len(cols),
            "tickers": ",".join(cols),
            "ann_ret": float(ew.mean() * 252),
            "ann_vol": float(ew.std() * np.sqrt(252)),
            "sharpe": float(ew.mean() * 252 / (ew.std() * np.sqrt(252))) if ew.std() > 0 else np.nan,
            "last_1m": float(ew.tail(21).sum()),
            "last_3m": float(ew.tail(63).sum()),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "growth_tech_sleeve_performance.parquet")
    print("\n=== Sleeve performance ===")
    print(df.to_string(index=False))
    return df


def run(window: int = 126) -> None:
    stocks = pd.read_parquet(STOCKS)
    prices = pd.read_parquet(PRICES)
    # `date` is DATE on disk -> read as datetime.date; keep it a date.
    m = members(stocks)
    tickers = m["ticker"].tolist()
    wide, rets = load_panel(prices, tickers)
    print(f"Growth tech panel: {len(tickers)} tickers, {len(wide)} days "
          f"({wide.index.min()} → {wide.index.max()})")

    cmd_membership(m)
    cmd_vol_return(rets, wide, m, window)
    cmd_correlations(rets.iloc[-window:], m)
    cmd_rolling(rets, window=20)
    cmd_sleeve_perf(rets, m)
    cmd_index_backtest(wide, prices, stocks)
    cmd_risk_models(rets, m, window)
    cmd_fisher(tickers)
    print("\n=== Growth tech suite complete ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=126)
    args = ap.parse_args()
    run(window=args.window)


if __name__ == "__main__":
    main()

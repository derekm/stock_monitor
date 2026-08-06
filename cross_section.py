#!/usr/bin/env python3
"""
cross_section.py — Multi-factor cross-section: rank the universe on
value + quality + momentum, long top quintile / short bottom quintile,
monthly rebalance, sector-neutral.

Method (point-in-time, OOS by construction):
  - At EACH rebalance date, factors are computed from data available then:
      value    = -z(pb_ratio)                         from fundamentals.parquet
                 (most recent row with as_of_date <= rebalance date)
      quality  =  z(roe) + z(roic) - z(debt_to_equity)  same as-of lookup
      momentum =  z(mom_12_1) + z(ret_21d)           from price history up to
                                                      rebalance date (no future)
  - Sector-neutral rank: within each sector, percentile-rank each factor,
    average the ranks, then take top/bottom quintile OF THE AVERAGE RANK.
  - Monthly rebalance on month-end dates; hold until next rebalance.
  - Baseline: equal-weight long-only top quintile (no short), for contrast.
  - Stats reported via cv_utils.oos_stats_vs_baseline (L/S vs baseline).

Outputs:
  cross_section_rankings.csv   per ticker per rebalance: ranks, bucket L/S
  cross_section_returns.csv    daily portfolio returns (long, short, L/S, EW)
  cross_section_stats.csv      OOS stats vs baseline + sector exposure check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns
from cv_utils import oos_stats_vs_baseline
from cost_model import apply_costs_to_daily

FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_RANK = DATA_DIR / "cross_section_rankings.csv"
OUT_RET = DATA_DIR / "cross_section_returns.csv"
OUT_STATS = DATA_DIR / "cross_section_stats.csv"


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if not sd or not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def _asof_fundamentals(rebal_date: pd.Timestamp) -> pd.DataFrame:
    """Value/quality factors as of a rebalance date (no future data)."""
    fund = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    if fund.empty or "as_of_date" not in fund.columns:
        return pd.DataFrame(columns=["ticker", "value_z", "quality_z"])
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"], errors="coerce")
    fund = fund[fund["as_of_date"] <= rebal_date]
    if fund.empty:
        return pd.DataFrame(columns=["ticker", "value_z", "quality_z"])
    fund = fund.sort_values("as_of_date").groupby("ticker").tail(1)
    df = fund[["ticker", "pb_ratio", "roe", "roic", "debt_to_equity"]].copy()
    df = df.dropna(subset=["pb_ratio", "roe", "roic"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "value_z", "quality_z"])
    df["value_z"] = _z(-df["pb_ratio"])
    df["quality_z"] = (
        _z(df["roe"]).fillna(0) + _z(df["roic"]).fillna(0) - _z(df["debt_to_equity"]).fillna(0)
    )
    return df[["ticker", "value_z", "quality_z"]]


def _momentum_factors(wide: pd.DataFrame, rebal_date: pd.Timestamp) -> pd.DataFrame:
    """Trailing momentum at a rebalance date (no future data)."""
    idx = wide.index[wide.index <= rebal_date]
    if len(idx) < 260:
        return pd.DataFrame(columns=["ticker", "momentum_z"])
    w = wide.loc[idx]
    t = idx[-1]
    pos21 = wide.index.get_indexer([t], method="nearest")[0]
    # mom_12_1 = ret from t-252 to t-21 ; ret_21d = ret over last 21 days
    mom = {}
    for tk in w.columns:
        s = w[tk].dropna()
        if len(s) < 260:
            continue
        p_now = s.iloc[-1]
        p_21 = s.iloc[-21] if len(s) >= 21 else np.nan
        p_252 = s.iloc[-252] if len(s) >= 252 else np.nan
        r21 = p_now / p_21 - 1 if p_21 and p_21 > 0 else np.nan
        mom_12_1 = p_21 / p_252 - 1 if p_21 and p_252 and p_252 > 0 else np.nan
        mom[tk] = (mom_12_1, r21)
    df = pd.DataFrame.from_dict(mom, orient="index", columns=["mom_12_1", "ret_21d"]).reset_index()
    df = df.rename(columns={"index": "ticker"})
    df["momentum_z"] = _z(df["mom_12_1"]).fillna(0) + _z(df["ret_21d"]).fillna(0)
    return df[["ticker", "momentum_z"]]


def _sector_map() -> dict[str, str]:
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    if stocks.empty or "ticker" not in stocks.columns:
        return {}
    out = {}
    for _, r in stocks.iterrows():
        out[str(r["ticker"]).upper()] = str(r.get("sector") or "unknown")
    return out


def sector_neutral_rank(f: pd.DataFrame) -> pd.DataFrame:
    """Within-sector percentile ranks, averaged; adds rank_avg + bucket."""
    if f.empty:
        return f
    f = f.copy()
    f["rank_value"] = f.groupby("sector")["value_z"].rank(pct=True)
    f["rank_quality"] = f.groupby("sector")["quality_z"].rank(pct=True)
    f["rank_momentum"] = f.groupby("sector")["momentum_z"].rank(pct=True)
    f["rank_avg"] = f[["rank_value", "rank_quality", "rank_momentum"]].mean(axis=1)
    f["bucket"] = pd.qcut(f["rank_avg"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    f.loc[f["rank_avg"].isna(), "bucket"] = np.nan
    return f


def monthly_rebalance_dates(wide: pd.DataFrame) -> list[pd.Timestamp]:
    idx = wide.index
    months = idx.to_period("M").unique()
    return [sub[-1] for m in months if len(sub := idx[idx.to_period("M") == m])]


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stocks = pd.read_parquet(STOCKS) if STOCKS.exists() else pd.DataFrame()
    tickers = sorted(stocks["ticker"].astype(str).str.upper().unique()) if not stocks.empty else []
    sector_map = _sector_map()

    prices = load_adj_prices_pandas(tickers=tickers)
    wide = wide_closes(prices).sort_index().dropna(how="all")
    # restrict to the window where factor data exists (fundamentals as_of range)
    fund = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    if not fund.empty and "as_of_date" in fund.columns:
        fund["as_of_date"] = pd.to_datetime(fund["as_of_date"], errors="coerce")
        lo = fund["as_of_date"].min()
        wide = wide[wide.index >= lo]
    rebal = monthly_rebalance_dates(wide)
    if len(rebal) < 6:
        raise SystemExit("Too few monthly rebalance dates in price history")

    rets = clip_returns(wide.pct_change(), 0.35)
    daily_idx = rets.index
    long_ret = pd.Series(0.0, index=daily_idx)
    short_ret = pd.Series(0.0, index=daily_idx)
    ls_ret = pd.Series(0.0, index=daily_idx)
    ew_ret = pd.Series(0.0, index=daily_idx)
    all_rankings: list[dict] = []
    exposure_devs: list[float] = []

    for i, rb in enumerate(rebal):
        nxt = rebal[i + 1] if i + 1 < len(rebal) else None
        hold = daily_idx[daily_idx > rb] if nxt is None else daily_idx[(daily_idx > rb) & (daily_idx <= nxt)]
        if len(hold) == 0:
            continue
        # point-in-time factors at this rebalance
        vq = _asof_fundamentals(rb)
        mom = _momentum_factors(wide, rb)
        if vq.empty or mom.empty:
            continue
        f = vq.merge(mom, on="ticker", how="outer")
        f["sector"] = f["ticker"].map(sector_map).fillna("unknown")
        f = sector_neutral_rank(f)
        longs = f[f["bucket"] == 5]["ticker"].tolist()
        shorts = f[f["bucket"] == 1]["ticker"].tolist()
        longs = [t for t in longs if t in rets.columns]
        shorts = [t for t in shorts if t in rets.columns]
        for t in longs:
            all_rankings.append({"rebalance_date": rb.date(), "ticker": t, "bucket": 5})
        for t in shorts:
            all_rankings.append({"rebalance_date": rb.date(), "ticker": t, "bucket": 1})
        if longs:
            long_ret.loc[hold] += rets.loc[hold, longs].mean(axis=1)
            ew_ret.loc[hold] += rets.loc[hold, longs].mean(axis=1)
        if shorts:
            short_ret.loc[hold] += -rets.loc[hold, shorts].mean(axis=1)
        ls_ret.loc[hold] = long_ret.loc[hold] + short_ret.loc[hold]
        # sector exposure deviation vs universe
        if not f.empty and "sector" in f.columns:
            univ_w = f.groupby("sector")["ticker"].count() / len(f)
            long_w = f[f["bucket"] == 5].groupby("sector")["ticker"].count()
            if len(long_w):
                exposure_devs.append(float((long_w / long_w.sum() - univ_w).abs().mean()))

    out = pd.DataFrame({
        "long": long_ret,
        "short": short_ret,
        "long_short": ls_ret,
        "equal_weight_long": ew_ret,
    })
    # net of costs: monthly rebalance = 2 × 10bps per month ≈ 20bps/21d turnover
    out = apply_costs_to_daily(out, turnover_frac=1.0 / 21.0)
    rankings = pd.DataFrame(all_rankings)
    stats = oos_stats_vs_baseline(out["long_short"], out["equal_weight_long"])
    if exposure_devs:
        stats["sector_exposure_abs_dev_avg"] = round(float(np.mean(exposure_devs)), 4)
    return rankings, out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    rankings, rets, stats = build()
    print("=== OOS stats (L/S vs equal-weight long baseline) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if len(rankings):
        print(f"\n=== rebalance periods: {rankings['rebalance_date'].nunique()} | long+short picks: {len(rankings)} ===")
        last = rankings[rankings["rebalance_date"] == rankings["rebalance_date"].max()]
        print("last rebalance long picks:")
        print(last[last["bucket"] == 5].head(10).to_string(index=False))
        print("last rebalance short picks:")
        print(last[last["bucket"] == 1].head(10).to_string(index=False))
    if args.save:
        rankings.to_csv(OUT_RANK, index=False)
        rets.to_csv(OUT_RET, index=False)
        pd.DataFrame([stats]).to_csv(OUT_STATS, index=False)
        print(f"\nWrote {OUT_RANK}\nWrote {OUT_RET}\nWrote {OUT_STATS}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
factor_library.py — Unified factor computation for StockMonitor.

Computes Fama-French 5 factors + Momentum (FF5+MOM) on our daily_prices universe,
plus Novy-Marx quality factors (gross profitability, investment, accruals).

All factors are computed daily, aligned to our trading calendar, and saved as parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent


def load_prices() -> pd.DataFrame:
    """Load daily prices, return close pivot (date × ticker)."""
    prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet")
    # Ensure date is datetime.date
    if prices["date"].dtype != "datetime64[ns]":
        prices["date"] = pd.to_datetime(prices["date"]).dt.date
    # Deduplicate: keep last entry per (date, ticker)
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    # Pivot to wide: date index, ticker columns, close values
    close = prices.pivot(index="date", columns="ticker", values="close")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    # Forward-fill missing (max 5 days) then drop all-NaN columns
    close = close.ffill(limit=5).dropna(axis=1, how="all")
    return close


def load_fundamentals() -> pd.DataFrame:
    """Load fundamentals, return latest quarterly per ticker."""
    fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    # Ensure as_of_date is datetime.date
    if "as_of_date" in fund.columns:
        if fund["as_of_date"].dtype != "datetime64[ns]":
            fund["as_of_date"] = pd.to_datetime(fund["as_of_date"]).dt.date
        # Rename for consistency
        fund = fund.rename(columns={"as_of_date": "date"})
    return fund


def compute_market_cap(close: pd.DataFrame, shares_outstanding: pd.Series | None = None) -> pd.DataFrame:
    """Market cap = close × shares_outstanding. If shares not provided, use close as proxy."""
    if shares_outstanding is not None:
        # Align shares to close columns
        shares = shares_outstanding.reindex(close.columns).fillna(1.0)
        mktcap = close.mul(shares, axis=1)
    else:
        # Proxy: use close price (relative size only)
        mktcap = close.copy()
    return mktcap


def compute_ff5(close: pd.DataFrame, mktcap: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Compute Fama-French 5 factors + Momentum on our universe.

    Factors:
    - MKT: market excess return (value-weighted market - risk-free)
    - SMB: small minus big (size)
    - HML: high minus low (value)
    - RMW: robust minus weak (profitability)
    - CMA: conservative minus aggressive (investment)
    - MOM: momentum (12-1)

    Returns DataFrame with date index and factor columns.
    """
    # Daily returns
    rets = close.pct_change()
    rets = rets.dropna(how="all")

    # Market cap weights (lagged 1 day to avoid look-ahead)
    mktcap_lagged = mktcap.shift(1).reindex(rets.index).ffill()
    mktcap_weights = mktcap_lagged.div(mktcap_lagged.sum(axis=1), axis=0)

    # Risk-free rate proxy: 0 for now (can plug in T-bill later)
    rf = 0.0

    # Market excess return (value-weighted)
    mkt_ret = (rets * mktcap_weights).sum(axis=1) - rf

    # ---- SMB / HML / RMW / CMA ----
    # Need fundamentals for B/M, profitability, investment
    # For now, compute size (SMB) and momentum (MOM) from prices only
    # Full FF5 needs fundamentals — see compute_ff5_with_fundamentals()

    # Size: median market cap split
    median_cap = mktcap_lagged.median(axis=1)
    small_mask = mktcap_lagged.lt(median_cap, axis=0)
    big_mask = ~small_mask

    # Equal-weight within size buckets
    small_weights = small_mask.div(small_mask.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    big_weights = big_mask.div(big_mask.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)

    smb = (rets * small_weights).sum(axis=1) - (rets * big_weights).sum(axis=1)

    # Momentum: 12-month return skipping last month (11/1/0)
    mom_lookback = 252  # ~12 months
    mom_skip = 21       # ~1 month
    mom_rets = close.pct_change(mom_lookback).shift(mom_skip)
    mom_rets = mom_rets.reindex(rets.index).ffill()

    # Top 30% minus bottom 30% by momentum
    mom_rank = mom_rets.rank(axis=1, pct=True)
    mom_long = mom_rank >= 0.7
    mom_short = mom_rank <= 0.3
    mom_long_w = mom_long.div(mom_long.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    mom_short_w = mom_short.div(mom_short.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    mom = (rets * mom_long_w).sum(axis=1) - (rets * mom_short_w).sum(axis=1)

    # Assemble factors
    factors = pd.DataFrame({
        "MKT": mkt_ret,
        "SMB": smb,
        "MOM": mom,
    }, index=rets.index)

    return factors


def compute_ff5_with_fundamentals(
    close: pd.DataFrame,
    mktcap: pd.DataFrame,
    fundamentals: pd.DataFrame,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Full FF5 + MOM using fundamentals for HML, RMW, CMA.

    Requires fundamentals columns (quarterly):
    - book_equity (or total_equity)
    - gross_profit (or revenue - cogs)
    - total_assets
    - capex (or change in PPE)
    """
    # Deduplicate fundamentals
    fundamentals = fundamentals.drop_duplicates(subset=["date", "ticker"], keep="last")
    
    rets = close.pct_change().dropna(how="all")

    # Market cap weights (lagged)
    mktcap_lagged = mktcap.shift(1).reindex(rets.index).ffill()
    mktcap_weights = mktcap_lagged.div(mktcap_lagged.sum(axis=1), axis=0)

    rf = 0.0
    mkt_ret = (rets * mktcap_weights).sum(axis=1) - rf

    # --- Prepare fundamental signals (quarterly, forward-filled to daily) ---
    # We need: book_to_market, gross_profitability, asset_growth
    # fundamentals has columns like: ticker, date, book_equity, gross_profit, total_assets, etc.

    # Pivot using the panel's actual column names
    fund_pivots = {}
    colmap = {
        "book_equity": "shareholders_equity",
        "gross_profit": "gross_profit" if "gross_profit" in fundamentals.columns else "revenue_ttm",
        "total_assets": "total_assets",
    }
    cal = pd.DatetimeIndex(pd.to_datetime(rets.index))
    for metric, col in colmap.items():
        piv = _pivot_fund(fundamentals, col, cal)
        if not piv.empty:
            fund_pivots[metric] = piv.reindex(index=rets.index, columns=rets.columns)

    # Book-to-market = book_equity / market_cap (lagged)
    if "book_equity" in fund_pivots:
        bm = fund_pivots["book_equity"].div(mktcap_lagged.replace(0, np.nan))
        bm = bm.replace([np.inf, -np.inf], np.nan)
    else:
        bm = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)

    # Gross profitability = gross_profit / total_assets
    if "gross_profit" in fund_pivots and "total_assets" in fund_pivots:
        gp = fund_pivots["gross_profit"].div(fund_pivots["total_assets"].replace(0, np.nan))
        gp = gp.replace([np.inf, -np.inf], np.nan)
    else:
        gp = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)

    # Asset growth from consecutive filings (not daily ffill of levels)
    ag = _asset_growth_from_filings(fundamentals, pd.DatetimeIndex(rets.index))
    ag = ag.reindex(index=rets.index, columns=rets.columns)

    # --- 2x3 sorts (size × value/profitability/investment) ---
    # Size breakpoint: median market cap
    median_cap = mktcap_lagged.median(axis=1)
    small = mktcap_lagged.lt(median_cap, axis=0)
    big = ~small

    def value_weighted_return(rets_df: pd.DataFrame, mask_df: pd.DataFrame, mktcap_w: pd.DataFrame) -> pd.Series:
        """Value-weighted return within a mask."""
        w = mktcap_w * mask_df
        w = w.div(w.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
        return (rets_df * w).sum(axis=1)

    # HML: High B/M minus Low B/M (within size buckets)
    if not bm.isna().all().all():
        bm_rank = bm.rank(axis=1, pct=True)
        high_bm = bm_rank >= 0.7
        low_bm = bm_rank <= 0.3

        hml_small = value_weighted_return(rets, small & high_bm, mktcap_weights) - \
                    value_weighted_return(rets, small & low_bm, mktcap_weights)
        hml_big = value_weighted_return(rets, big & high_bm, mktcap_weights) - \
                  value_weighted_return(rets, big & low_bm, mktcap_weights)
        hml = (hml_small + hml_big) / 2
    else:
        hml = pd.Series(0.0, index=rets.index)

    # RMW: Robust (high profitability) minus Weak (low profitability)
    if not gp.isna().all().all():
        gp_rank = gp.rank(axis=1, pct=True)
        high_gp = gp_rank >= 0.7
        low_gp = gp_rank <= 0.3

        rmw_small = value_weighted_return(rets, small & high_gp, mktcap_weights) - \
                    value_weighted_return(rets, small & low_gp, mktcap_weights)
        rmw_big = value_weighted_return(rets, big & high_gp, mktcap_weights) - \
                  value_weighted_return(rets, big & low_gp, mktcap_weights)
        rmw = (rmw_small + rmw_big) / 2
    else:
        rmw = pd.Series(0.0, index=rets.index)

    # CMA: Conservative (low investment) minus Aggressive (high investment)
    if not ag.isna().all().all():
        ag_rank = ag.rank(axis=1, pct=True)
        low_ag = ag_rank <= 0.3   # conservative
        high_ag = ag_rank >= 0.7  # aggressive

        cma_small = value_weighted_return(rets, small & low_ag, mktcap_weights) - \
                    value_weighted_return(rets, small & high_ag, mktcap_weights)
        cma_big = value_weighted_return(rets, big & low_ag, mktcap_weights) - \
                  value_weighted_return(rets, big & high_ag, mktcap_weights)
        cma = (cma_small + cma_big) / 2
    else:
        cma = pd.Series(0.0, index=rets.index)

    # Momentum (12-1)
    mom_lookback = 252
    mom_skip = 21
    mom_rets = close.pct_change(mom_lookback).shift(mom_skip).reindex(rets.index).ffill()
    mom_rank = mom_rets.rank(axis=1, pct=True)
    mom_long = mom_rank >= 0.7
    mom_short = mom_rank <= 0.3
    mom_long_w = mom_long.div(mom_long.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    mom_short_w = mom_short.div(mom_short.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    mom = (rets * mom_long_w).sum(axis=1) - (rets * mom_short_w).sum(axis=1)

    # Assemble
    factors = pd.DataFrame({
        "MKT": mkt_ret,
        "SMB": (value_weighted_return(rets, small, mktcap_weights) - value_weighted_return(rets, big, mktcap_weights)),
        "HML": hml,
        "RMW": rmw,
        "CMA": cma,
        "MOM": mom,
    }, index=rets.index)

    return factors


def _pivot_fund(fund: pd.DataFrame, col: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if col not in fund.columns:
        return pd.DataFrame(index=calendar)
    d = fund.dropna(subset=["date", "ticker", col]).drop_duplicates(subset=["date", "ticker"], keep="last")
    d = d.copy()
    d["ticker"] = d["ticker"].astype(str).str.upper()
    piv = d.pivot(index="date", columns="ticker", values=col)
    piv.index = pd.to_datetime(piv.index)
    return piv.sort_index().ffill().reindex(calendar).ffill()


def _asset_growth_from_filings(fund: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """% change in total_assets between consecutive filings, ffilled onto calendar.

    Defined only when a ticker has a prior total_assets observation.
    """
    if "total_assets" not in fund.columns:
        return pd.DataFrame(index=calendar)
    d = fund.dropna(subset=["date", "ticker", "total_assets"]).drop_duplicates(
        subset=["date", "ticker"], keep="last"
    ).copy()
    d["ticker"] = d["ticker"].astype(str).str.upper()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["ticker", "date"])
    d["prev_assets"] = d.groupby("ticker")["total_assets"].shift(1)
    d["asset_growth"] = (d["total_assets"] / d["prev_assets"].replace(0, np.nan)) - 1.0
    d.loc[d["prev_assets"].isna(), "asset_growth"] = np.nan
    d["asset_growth"] = d["asset_growth"].replace([np.inf, -np.inf], np.nan)
    g = d.dropna(subset=["asset_growth"]).pivot(index="date", columns="ticker", values="asset_growth")
    if g.empty:
        return pd.DataFrame(index=calendar)
    g.index = pd.to_datetime(g.index)
    return g.sort_index().ffill().reindex(calendar).ffill()


def compute_novymarx_quality(fundamentals: pd.DataFrame, mktcap: pd.DataFrame, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Novy-Marx quality panels (date × ticker):
    - gross_profitability: revenue_ttm / total_assets (GP proxy; no COGS panel)
    - asset_growth: consecutive-filing % change in total_assets
    - accruals: (net_income_ttm - operating_cash_flow_ttm) / total_assets
    - debt_to_equity: total_debt / shareholders_equity
    - book_to_market: shareholders_equity / market_cap (fundamentals mcap, else price mcap)
    """
    fundamentals = fundamentals.drop_duplicates(subset=["date", "ticker"], keep="last")
    calendar = pd.DatetimeIndex(pd.to_datetime(close.index))
    quality: dict[str, pd.DataFrame] = {}

    rev = _pivot_fund(fundamentals, "revenue_ttm", calendar)
    assets = _pivot_fund(fundamentals, "total_assets", calendar)
    ni = _pivot_fund(fundamentals, "net_income_ttm", calendar)
    ocf = _pivot_fund(fundamentals, "operating_cash_flow_ttm", calendar)
    debt = _pivot_fund(fundamentals, "total_debt", calendar)
    equity = _pivot_fund(fundamentals, "shareholders_equity", calendar)
    fund_mcap = _pivot_fund(fundamentals, "market_cap", calendar)

    gp_num = _pivot_fund(fundamentals, "gross_profit", calendar)
    if gp_num.empty or not gp_num.notna().any().any():
        gp_num = rev
    if not gp_num.empty and not assets.empty:
        quality["gross_profitability"] = gp_num.div(assets.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    quality["asset_growth"] = _asset_growth_from_filings(fundamentals, calendar)

    if not ni.empty and not ocf.empty and not assets.empty:
        quality["accruals"] = (ni - ocf).div(assets.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if not debt.empty and not equity.empty:
        quality["debt_to_equity"] = debt.div(equity.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if not equity.empty:
        if not fund_mcap.empty and fund_mcap.notna().any().any():
            m = fund_mcap.reindex(index=calendar, columns=equity.columns).ffill()
        else:
            m = mktcap.reindex(index=calendar).ffill().reindex(columns=equity.columns).ffill()
        quality["book_to_market"] = equity.div(m.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    return quality


AG_WINSOR = 1.0  # clip filing %Δ at ±100% for ranking only


def attach_nm_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-filing NM ranks and nm_quality. Does not change buffett_pass."""
    out = df.copy()
    if "ticker" not in out.columns:
        return out
    out["ticker"] = out["ticker"].astype(str).str.upper()

    def last_nn(name: str) -> pd.Series:
        path = DATA_DIR / f"novymarx_{name}.parquet"
        if not path.exists():
            return pd.Series(dtype=float)
        panel = pd.read_parquet(path)
        panel.columns = panel.columns.astype(str).str.upper()
        return panel.ffill().iloc[-1]

    nm = pd.DataFrame({
        "nm_gross_profitability": last_nn("gross_profitability"),
        "nm_asset_growth": last_nn("asset_growth"),
        "nm_accruals": last_nn("accruals"),
        "nm_debt_to_equity": last_nn("debt_to_equity"),
        "nm_book_to_market": last_nn("book_to_market"),
    })
    ag = nm["nm_asset_growth"].clip(-AG_WINSOR, AG_WINSOR)
    nm["gp_q"] = nm["nm_gross_profitability"].rank(pct=True)
    nm["ag_q"] = (1 - ag.rank(pct=True))
    nm["ac_q"] = 1 - nm["nm_accruals"].rank(pct=True)
    nm["de_q"] = 1 - nm["nm_debt_to_equity"].rank(pct=True)
    nm["nm_score"] = nm[["gp_q", "ag_q", "ac_q", "de_q"]].mean(axis=1)
    nm["nm_legs"] = nm[["gp_q", "ag_q", "ac_q", "de_q"]].notna().sum(axis=1)
    nm["nm_quality"] = (nm["nm_score"] >= 0.5) & (nm["nm_legs"] >= 2)
    nm.index.name = "ticker"
    nm = nm.reset_index()
    keep = ["ticker", "nm_gross_profitability", "nm_asset_growth", "nm_accruals",
            "nm_debt_to_equity", "nm_book_to_market", "nm_score", "nm_legs", "nm_quality"]
    out = out.drop(columns=[c for c in keep if c != "ticker" and c in out.columns], errors="ignore")
    return out.merge(nm[keep], on="ticker", how="left")


def main():
    ap = argparse.ArgumentParser(description="Compute factor library")
    ap.add_argument("--save", action="store_true", help="Save outputs to parquet")
    ap.add_argument("--full", action="store_true", help="Compute full FF5 with fundamentals (slower)")
    ap.add_argument("--quality-only", action="store_true",
                    help="Novy-Marx panels only; no daily_prices read")
    args = ap.parse_args()

    if args.quality_only:
        print("Loading fundamentals...")
        import shutil
        import tempfile
        snap = Path(tempfile.gettempdir()) / "fl_fundamentals.parquet"
        shutil.copy2(DATA_DIR / "fundamentals.parquet", snap)
        fund = pd.read_parquet(snap)
        if "as_of_date" in fund.columns:
            fund = fund.rename(columns={"as_of_date": "date"})
        dates = pd.to_datetime(fund["date"], errors="coerce").dropna().drop_duplicates().sort_values()
        calendar = pd.DatetimeIndex(dates)
        print(f"  filing calendar: {len(calendar)} dates")
        close = pd.DataFrame(index=calendar)
        mktcap = pd.DataFrame(index=calendar)
        print("Computing Novy-Marx quality...")
        quality_dict = compute_novymarx_quality(fund, mktcap, close)
        if args.save:
            for metric_name, metric_df in quality_dict.items():
                quality_path = DATA_DIR / f"novymarx_{metric_name}.parquet"
                metric_df.to_parquet(quality_path)
                print(f"Saved {quality_path} ({len(metric_df)} rows × {metric_df.shape[1]} tickers)")
        print(f"\nQuality metrics available: {list(quality_dict.keys())}")
        for metric_name, metric_df in quality_dict.items():
            last = metric_df.iloc[-1] if len(metric_df) else pd.Series(dtype=float)
            print(f"  {metric_name}: latest non-null {int(last.notna().sum()):,} tickers; "
                  f"panel {metric_df.notna().sum().sum():,} / {metric_df.size:,}")
        return None, quality_dict

    print("Loading prices...")
    close = load_prices()
    print(f"  {close.shape[0]} dates × {close.shape[1]} tickers")

    print("Loading fundamentals...")
    fundamentals = load_fundamentals()

    print("Computing market cap...")
    mktcap = compute_market_cap(close)

    if args.full:
        print("Computing full FF5 + MOM with fundamentals...")
        factors = compute_ff5_with_fundamentals(close, mktcap, fundamentals)
    else:
        print("Computing FF5+MOM (price-only: MKT, SMB, MOM)...")
        factors = compute_ff5(close, mktcap)

    print("Computing Novy-Marx quality...")
    quality_dict = compute_novymarx_quality(fundamentals, mktcap, close)

    if args.save:
        factors_path = DATA_DIR / "ff5_factors.parquet"
        # Save each quality metric as separate parquet
        for metric_name, metric_df in quality_dict.items():
            quality_path = DATA_DIR / f"novymarx_{metric_name}.parquet"
            metric_df.to_parquet(quality_path)
            print(f"Saved {quality_path} ({len(metric_df)} rows × {metric_df.shape[1]} tickers)")
        factors.to_parquet(factors_path)
        print(f"Saved {factors_path} ({len(factors)} rows)")

    # Summary stats
    print("\n=== Factor Summary ===")
    print(factors.describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\nAnnualized MKT: {factors['MKT'].mean()*252:.2%}, vol: {factors['MKT'].std()*np.sqrt(252):.2%}")
    print(f"Annualized SMB: {factors['SMB'].mean()*252:.2%}, vol: {factors['SMB'].std()*np.sqrt(252):.2%}")
    if "HML" in factors.columns:
        print(f"Annualized HML: {factors['HML'].mean()*252:.2%}, vol: {factors['HML'].std()*np.sqrt(252):.2%}")
    if "RMW" in factors.columns:
        print(f"Annualized RMW: {factors['RMW'].mean()*252:.2%}, vol: {factors['RMW'].std()*np.sqrt(252):.2%}")
    if "CMA" in factors.columns:
        print(f"Annualized CMA: {factors['CMA'].mean()*252:.2%}, vol: {factors['CMA'].std()*np.sqrt(252):.2%}")
    print(f"Annualized MOM: {factors['MOM'].mean()*252:.2%}, vol: {factors['MOM'].std()*np.sqrt(252):.2%}")

    print(f"\nQuality metrics available: {list(quality_dict.keys())}")
    for metric_name, metric_df in quality_dict.items():
        print(f"  {metric_name}: {metric_df.notna().sum().sum():,} non-null / {metric_df.size:,} total")

    return factors, quality_dict


if __name__ == "__main__":
    main()
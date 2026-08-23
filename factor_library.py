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
from analytics_common import winsor_abs, winsor_cs

DATA_DIR = Path(__file__).parent


def load_prices() -> pd.DataFrame:
    """Load daily adj close, snapshot first (Windows lock)."""
    import shutil, tempfile
    snap = Path(tempfile.gettempdir()) / "fl_daily_prices.parquet"
    shutil.copy2(DATA_DIR / "daily_prices.parquet", snap)
    prices = pd.read_parquet(snap, columns=["date", "ticker", "adj_close", "close"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    px = prices["adj_close"].where(prices["adj_close"].notna(), prices["close"])
    close = prices.assign(px=px).pivot(index="date", columns="ticker", values="px")
    close.index = pd.to_datetime(close.index)
    return close.sort_index().ffill(limit=5).dropna(axis=1, how="all")


def load_fundamentals() -> pd.DataFrame:
    """Load fundamentals snapshot."""
    import shutil, tempfile
    snap = Path(tempfile.gettempdir()) / "fl_fundamentals.parquet"
    shutil.copy2(DATA_DIR / "fundamentals.parquet", snap)
    fund = pd.read_parquet(snap)
    if "as_of_date" in fund.columns:
        fund = fund.rename(columns={"as_of_date": "date"})
    fund["ticker"] = fund["ticker"].astype(str).str.upper()
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
    # Drop data-error prints from the VW book; keep them out of weights too.
    bad = rets.abs().gt(0.20)
    rets = rets.mask(bad)

    mktcap_lagged = mktcap.shift(1).reindex(rets.index)
    # Winsorize per date so a share-count blowup cannot dominate VW.
    mktcap_lagged = winsor_cs(mktcap_lagged, 0.995)
    mktcap_lagged = mktcap_lagged.where(mktcap_lagged.gt(0))
    w = mktcap_lagged.where(rets.notna())
    mktcap_weights = w.div(w.sum(axis=1), axis=0)
    mkt_ret = (rets * mktcap_weights).sum(axis=1, min_count=50)

    # --- Prepare fundamental signals (quarterly, forward-filled to daily) ---
    # We need: book_to_market, gross_profitability, asset_growth
    # fundamentals has columns like: ticker, date, book_equity, gross_profit, total_assets, etc.

    # Pivot using the panel's actual column names
    fund_pivots = {}
    # Profitability: GP, else revenue − cogs, else revenue (Rev/A). Empty
    # `gross_profit` column must not hide the fallback — that zeroed RMW.
    gp = pd.to_numeric(fundamentals["gross_profit"], errors="coerce") if "gross_profit" in fundamentals.columns else None
    rev = pd.to_numeric(fundamentals["revenue_ttm"], errors="coerce") if "revenue_ttm" in fundamentals.columns else None
    cogs = pd.to_numeric(fundamentals["cogs"], errors="coerce") if "cogs" in fundamentals.columns else None
    prof = gp.copy() if gp is not None else pd.Series(np.nan, index=fundamentals.index)
    if rev is not None and cogs is not None:
        prof = prof.fillna(rev - cogs)
    if rev is not None:
        prof = prof.fillna(rev)
    fundamentals = fundamentals.copy()
    fundamentals["_profit"] = prof

    colmap = {
        "book_equity": "shareholders_equity",
        "gross_profit": "_profit",
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
    ag = winsor_abs(nm["nm_asset_growth"], AG_WINSOR)
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


def compute_regime_factor_premia() -> pd.DataFrame:
    """Ang-style regime-conditional means of available FF factors."""
    ff = pd.read_parquet(DATA_DIR / "ff5_factors.parquet")
    if "date" not in ff.columns:
        ff = ff.reset_index()
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()
    hmm = pd.read_parquet(DATA_DIR / "hmm_regime_states.parquet")
    hmm["date"] = pd.to_datetime(hmm["date"]).dt.normalize()
    hmm = hmm.drop_duplicates("date", keep="last")
    fac = [c for c in ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"] if c in ff.columns]
    m = ff.merge(hmm[["date", "regime"]], on="date", how="inner")
    rows = []
    for regime, g in m.groupby("regime"):
        row = {"regime": regime, "n_days": int(len(g))}
        for c in fac:
            s = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(s.mean())
            row[f"{c}_ann"] = float(s.mean() * 252)
            row[f"{c}_vol"] = float(s.std() * np.sqrt(252))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_days", ascending=False)
    return out


def residual_ic() -> pd.DataFrame:
    """PIT ER_{t-1} Spearman IC vs CAPM residual r_t − β̂_{t-1} MKT_t (fixed MKT)."""
    import tempfile
    er = pd.read_parquet(DATA_DIR / "expected_returns_decomp.parquet",
                         columns=["date", "ticker", "expected_return"])
    er = er.dropna(subset=["expected_return"])
    er["ticker"] = er["ticker"].astype(str).str.upper()
    er["month"] = pd.to_datetime(er["date"]).dt.to_period("M")
    names = er[er["month"] == er["month"].max()].nlargest(800, "expected_return")["ticker"].tolist()
    ff = pd.read_parquet(DATA_DIR / "ff5_factors.parquet")
    if "date" not in ff.columns:
        ff = ff.reset_index().rename(columns={ff.index.name or "index": "date"})
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()
    ff["month"] = ff["date"].dt.to_period("M")
    mkt = ff.groupby("month")["MKT"].sum()
    snap = Path(tempfile.gettempdir()) / "ph_daily_prices.parquet"
    px = pd.read_parquet(snap, columns=["date", "ticker", "adj_close", "close"])
    px["ticker"] = px["ticker"].astype(str).str.upper()
    px = px[px["ticker"].isin(names)]
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["px"] = px["adj_close"].where(px["adj_close"].notna(), px["close"])
    last = px.sort_values("date").groupby(["ticker", px["date"].dt.to_period("M")]).tail(1)
    last["month"] = last["date"].dt.to_period("M")
    rets = last.pivot(index="month", columns="ticker", values="px").pct_change()
    rets = winsor_abs(rets, 0.50)
    # trailing 36m beta vs MKT; apply to this month
    aligned = rets.reindex(mkt.index)
    ics = []
    months = list(aligned.index)
    for i, m in enumerate(months):
        if i < 36:
            continue
        win = aligned.iloc[i - 36:i]
        mk = mkt.reindex(win.index)
        if mk.notna().sum() < 24 or m not in mkt.index:
            continue
        y = aligned.loc[m].dropna()
        prev = er[er["month"] == months[i - 1]].drop_duplicates("ticker").set_index("ticker")["expected_return"]
        both = y.index.intersection(prev.index).intersection(win.columns)
        if len(both) < 40:
            continue
        # beta_i = cov(r_i, MKT) / var(MKT) on the window
        mkv = mk.to_numpy()
        v = np.nanvar(mkv)
        if not np.isfinite(v) or v < 1e-12:
            continue
        betas = {}
        for t in both:
            ri = win[t].to_numpy()
            ok = np.isfinite(ri) & np.isfinite(mkv)
            if ok.sum() < 24:
                continue
            betas[t] = float(np.cov(ri[ok], mkv[ok])[0, 1] / v)
        if len(betas) < 40:
            continue
        idx = pd.Index(betas.keys())
        resid = y.loc[idx] - pd.Series(betas) * float(mkt.loc[m])
        ic = resid.corr(prev.loc[idx], method="spearman")
        if np.isfinite(ic):
            ics.append({"month": str(m), "ic": float(ic), "n": int(len(idx))})
    out = pd.DataFrame(ics)
    print(out.tail(8).to_string(index=False) if len(out) else "no IC")
    if len(out):
        mu = float(out["ic"].mean())
        print(f"CAPM residual IC {mu:.4f}  n_months={len(out)}  bar +0.02  (fixed MKT)")
        out.to_parquet(DATA_DIR / "residual_ic.parquet", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute factor library")
    ap.add_argument("--save", action="store_true", help="Save outputs to parquet")
    ap.add_argument("--full", action="store_true", help="Compute full FF5 with fundamentals (slower)")
    ap.add_argument("--quality-only", action="store_true",
                    help="Novy-Marx panels only; no daily_prices read")
    ap.add_argument("--regime-premia", action="store_true",
                    help="Ang regime-conditional FF premia from hmm + ff5_factors")
    ap.add_argument("--hml", action="store_true",
                    help="Write HML/RMW/CMA onto ff5_factors (stock, 10y, snapshot)")
    ap.add_argument("--residual-ic", action="store_true",
                    help="Rank IC of nm_score vs returns residualized on HML/RMW/CMA/MOM (no MKT)")
    args = ap.parse_args()

    if args.residual_ic:
        return residual_ic(), None

    if args.hml:
        print("HML/RMW/CMA (stock, 10y, PIT mcap)...")
        close = load_prices()
        stocks = DATA_DIR / "monitored_stocks.parquet"
        if stocks.exists():
            ms = pd.read_parquet(stocks, columns=["ticker", "instrument_type"])
            keep = set(ms.loc[ms["instrument_type"].eq("stock"), "ticker"].astype(str).str.upper())
            close = close.reindex(columns=[c for c in close.columns if c in keep])
        cutoff = close.index.max() - pd.Timedelta(days=int(10 * 365.25))
        close = close.loc[close.index >= cutoff]
        print(f"  {close.shape[0]} dates × {close.shape[1]} tickers")
        panel_path = DATA_DIR / "daily_mcap.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            panel["ticker"] = panel["ticker"].astype(str).str.upper()
            panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
            mktcap = panel.pivot(index="date", columns="ticker", values="market_cap")
            close = close.copy()
            close.index = pd.to_datetime(close.index).normalize()
            close = close.groupby(level=0).last()
            mktcap = mktcap.reindex(index=close.index, columns=close.columns)
            print(f"  PIT mcap last nn {int(mktcap.iloc[-1].notna().sum()):,}")
            last_m = mktcap.iloc[-1]
            big = last_m.index[last_m.ge(1e9)]
            close = close.reindex(columns=big)
            mktcap = mktcap.reindex(columns=big)
            print(f"  mcap>=$1B: {len(big)} names")
        else:
            fund = load_fundamentals()
            shares = fund.dropna(subset=["ticker", "shares_outstanding"]).sort_values("date")
            sh = shares.groupby("ticker")["shares_outstanding"].last()
            mktcap = compute_market_cap(close, sh)
            print("  WARNING: daily_mcap.parquet missing — last-shares fallback")
        fund = load_fundamentals()
        factors = compute_ff5_with_fundamentals(close, mktcap, fund)
        if args.save:
            path = DATA_DIR / "ff5_factors.parquet"
            factors.to_parquet(path)
            print(f"Saved {path} {factors.shape} cols={list(factors.columns)}")
        print(factors.describe().T[["mean", "std"]].to_string())
        for c in factors.columns:
            print(f"  {c} ann {factors[c].mean()*252:.2%} vol {factors[c].std()*np.sqrt(252):.2%} nn {int(factors[c].notna().sum())}")
        return factors, None

    if args.regime_premia:
        premia = compute_regime_factor_premia()
        print(premia.to_string(index=False))
        if args.save:
            path = DATA_DIR / "regime_factor_premia.parquet"
            premia.to_parquet(path, index=False)
            print(f"Saved {path} ({len(premia)} regimes)")
        return premia, None

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
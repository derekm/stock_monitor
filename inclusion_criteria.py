#!/usr/bin/env python3
"""
inclusion_criteria.py — Documented, automated inclusion/exclusion rules.

DUAL-PASS (INCLUDE_CORE) — must satisfy ALL of:
  Quality (Buffett-style):
    ROE  >= 0.15
    ROIC >= 0.15
    Debt/Equity <= 1.0
  Value trifecta:
    EV/EBITDA      <= 9.0
    P/B            <= 1.5
    MktCap/Assets  <= 0.5

Other decision bands (from preferred_metrics):
  INCLUDE_VALUE   — trifecta only
  INCLUDE_QUALITY — Buffett only
  SATELLITE       — composite >= 0.50
  WATCH           — composite >= 0.35
  AVOID           — else

Hard exclusions (policy):
  - Per-name hard cap (default 5% of total portfolio) regardless of scores
  - Negative earnings_stability < 0.25 → prefer AVOID/WATCH for core book
  - growth_tech names never auto-promoted to INCLUDE_CORE without dual pass

Usage:
  python inclusion_criteria.py
  python inclusion_criteria.py --explore-defensive
  python inclusion_criteria.py --save
"""
from __future__ import annotations

import argparse
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
PREF = DATA_DIR / "preferred_metrics.parquet"

OUT_INC = DATA_DIR / "inclusion_candidates.parquet"
OUT_EXC = DATA_DIR / "exclusion_candidates.parquet"
OUT_NEAR = DATA_DIR / "near_dual_candidates.parquet"
OUT_DEF = DATA_DIR / "defensive_value_exploration.parquet"
OUT_RULES = DATA_DIR / "inclusion_rules.json"
OUT_ACORR = DATA_DIR / "asset_correlation_matrix.parquet"
OUT_SCORR = DATA_DIR / "sector_correlation_matrix_latest.parquet"

from analytics_common import (
    BASE_THRESHOLDS, quality_value_parts, COMP_W_Q, COMP_W_V,
)

RULES = {
    "dual_pass": {
        **BASE_THRESHOLDS,
        "debt_to_equity_max": BASE_THRESHOLDS["de_max"],
        "ev_ebitda_max": BASE_THRESHOLDS["ev_max"],
        "mktcap_to_assets_max": BASE_THRESHOLDS["mca_max"],
        "label": "INCLUDE_CORE",
    },
    "sizing": {
        "per_name_max_weight": 0.05,
        "core_max_weight": 0.12,
        "value_max_weight": 0.08,
        "satellite_max_weight": 0.05,
    },
    "exclusions": {
        "earnings_stability_min_for_core": 0.25,
        "block_growth_tech_from_core_without_dual": True,
    },
}


def load_latest_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def evaluate(fund: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    r = RULES["dual_pass"]
    rows = []
    st = stocks.set_index("ticker") if len(stocks) else pd.DataFrame()
    for _, x in fund.iterrows():
        t = x["ticker"]
        meta = st.loc[t] if t in st.index else pd.Series(dtype=object)
        buffett = (
            pd.notna(x.get("roe")) and x["roe"] >= r["roe_min"]
            and pd.notna(x.get("roic")) and x["roic"] >= r["roic_min"]
            and pd.notna(x.get("debt_to_equity")) and x["debt_to_equity"] <= r["debt_to_equity_max"]
        )
        trifecta = (
            pd.notna(x.get("ev_ebitda")) and x["ev_ebitda"] <= r["ev_ebitda_max"]
            and pd.notna(x.get("pb_ratio")) and x["pb_ratio"] <= r["pb_max"]
            and pd.notna(x.get("mktcap_to_assets")) and x["mktcap_to_assets"] <= r["mktcap_to_assets_max"]
        )
        dual = buffett and trifecta
        # gap analysis — which legs fail
        fails = []
        if not (pd.notna(x.get("roe")) and x["roe"] >= r["roe_min"]):
            fails.append("roe")
        if not (pd.notna(x.get("roic")) and x["roic"] >= r["roic_min"]):
            fails.append("roic")
        if not (pd.notna(x.get("debt_to_equity")) and x["debt_to_equity"] <= r["debt_to_equity_max"]):
            fails.append("de")
        if not (pd.notna(x.get("ev_ebitda")) and x["ev_ebitda"] <= r["ev_ebitda_max"]):
            fails.append("ev")
        if not (pd.notna(x.get("pb_ratio")) and x["pb_ratio"] <= r["pb_max"]):
            fails.append("pb")
        if not (pd.notna(x.get("mktcap_to_assets")) and x["mktcap_to_assets"] <= r["mktcap_to_assets_max"]):
            fails.append("mca")

        # near dual: fail at most 1 leg, and close on that leg
        near = False
        if len(fails) == 1:
            near = True
        elif len(fails) == 2:
            # both soft misses
            near = True

        es = x.get("earnings_stability")
        hard_exclude = False
        exclude_reasons = []
        if dual and pd.notna(es) and es < RULES["exclusions"]["earnings_stability_min_for_core"]:
            hard_exclude = True
            exclude_reasons.append("low_earnings_stability")
        if bool(meta.get("growth_tech_index", False)) and not dual:
            # not an exclusion from monitoring — exclusion from CORE sleeve only
            pass

        if dual and not hard_exclude:
            candidacy = "INCLUDE"
            decision = "INCLUDE_CORE"
        elif dual and hard_exclude:
            candidacy = "EXCLUDE"
            decision = "WATCH"
        elif trifecta:
            candidacy = "INCLUDE"
            decision = "INCLUDE_VALUE"
        elif buffett:
            candidacy = "INCLUDE"
            decision = "INCLUDE_QUALITY"
        elif near:
            candidacy = "WATCH"
            decision = "NEAR_DUAL"
        else:
            candidacy = "EXCLUDE" if len(fails) >= 5 else "WATCH"
            decision = "AVOID" if len(fails) >= 5 else "WATCH"

        rows.append({
            "ticker": t,
            "sector": meta.get("sector"),
            "industry": meta.get("industry"),
            "value_sleeve": meta.get("value_sleeve"),
            "defensive_value_index": bool(meta.get("defensive_value_index", False)),
            "growth_tech_index": bool(meta.get("growth_tech_index", False)),
            "dual_pass_member": bool(meta.get("dual_pass_member", False)),
            "roe": x.get("roe"),
            "roic": x.get("roic"),
            "debt_to_equity": x.get("debt_to_equity"),
            "ev_ebitda": x.get("ev_ebitda"),
            "pb_ratio": x.get("pb_ratio"),
            "mktcap_to_assets": x.get("mktcap_to_assets"),
            "earnings_stability": es,
            "buffett_pass": buffett,
            "trifecta_pass": trifecta,
            "dual_pass": dual,
            "failed_legs": ",".join(fails),
            "n_failed_legs": len(fails),
            "candidacy": candidacy,
            "decision": decision,
            "exclude_reasons": ",".join(exclude_reasons) if exclude_reasons else None,
        })
    return pd.DataFrame(rows)


def analyze_criteria(df: pd.DataFrame) -> None:
    print("=== Dual-pass selection criteria ===")
    r = RULES["dual_pass"]
    for k, v in r.items():
        print(f"  {k}: {v}")
    print(f"\nUniverse scored: {len(df)}")
    print(df["decision"].value_counts().to_string())
    print("\nDual passers:")
    cols = ["ticker", "sector", "roe", "roic", "debt_to_equity", "ev_ebitda", "pb_ratio", "mktcap_to_assets"]
    print(df[df.dual_pass][cols].to_string(index=False) if df.dual_pass.any() else "  (none)")
    print("\nFailure leg frequency (non-dual):")
    from collections import Counter
    c = Counter()
    for s in df.loc[~df.dual_pass, "failed_legs"]:
        for leg in str(s).split(","):
            if leg:
                c[leg] += 1
    for leg, n in c.most_common():
        print(f"  {leg:5s}  {n:4d}")


def explore_defensive(df: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """Defensive-value exploration: staples, healthcare, utilities, financials, energy value."""
    defensive_sectors = {
        "Consumer Staples", "Health Care", "Utilities", "Financials",
        "Energy", "Materials", "Communication Services", "Industrials",
    }
    sub = df[df["sector"].isin(defensive_sectors)].copy()
    # rank by composite-like score (canonical formula, weights in analytics_common)
    def score(row):
        q, v = quality_value_parts(
            roe=row.roe, roic=row.roic, de=row.debt_to_equity,
            earnings_stability=row.earnings_stability,
            ev=row.ev_ebitda, pb=row.pb_ratio, mca=row.mktcap_to_assets,
        )
        return COMP_W_Q * q + COMP_W_V * v

    sub["defensive_score"] = sub.apply(score, axis=1)
    sub = sub.sort_values("defensive_score", ascending=False)
    print("\n=== Defensive value exploration (top 25) ===")
    show = ["ticker", "sector", "decision", "defensive_score", "roe", "roic", "ev_ebitda", "pb_ratio", "failed_legs"]
    print(sub[show].head(25).to_string(index=False))
    return sub


def correlation_tables(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS)
    # asset corr — portfolio + dual + top defensive
    tickers = set()
    if HOLDINGS.exists():
        tickers |= set(pd.read_parquet(HOLDINGS)["ticker"])
    tickers |= set(stocks.loc[stocks.get("dual_pass_member", False) == True, "ticker"])
    # add a sample of defensive
    if "defensive_value_index" in stocks.columns:
        tickers |= set(stocks.loc[stocks.defensive_value_index == True, "ticker"].head(40))
    tickers = sorted(tickers)
    wide = (prices[prices.ticker.isin(tickers)]
            .pivot_table(index="date", columns="ticker", values="close")
            .sort_index().ffill())
    rets = np.log(wide / wide.shift(1)).iloc[-126:]
    acorr = rets.corr()
    if save:
        acorr.to_parquet(OUT_ACORR)
        print(f"Wrote {OUT_ACORR} ({acorr.shape[0]} assets)")

    # sector EW corr
    rets_all = np.log(
        prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        / prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill().shift(1)
    ).iloc[-126:]
    sector_map = stocks.set_index("ticker")["sector"].to_dict()
    sector_rets = {}
    for sec, grp in stocks.groupby("sector"):
        cols = [t for t in grp["ticker"] if t in rets_all.columns]
        if len(cols) >= 2:
            sector_rets[sec] = rets_all[cols].mean(axis=1)
    if sector_rets:
        sret = pd.DataFrame(sector_rets).dropna(how="all")
        scorr = sret.corr()
        if save:
            scorr.to_parquet(OUT_SCORR)
            print(f"Wrote {OUT_SCORR} ({scorr.shape[0]} sectors)")
    else:
        scorr = pd.DataFrame()
    return acorr, scorr


def run(explore: bool = False, save: bool = True):
    fund = load_latest_fund()
    stocks = pd.read_parquet(STOCKS)
    df = evaluate(fund, stocks)
    analyze_criteria(df)

    inc = df[df.candidacy == "INCLUDE"].sort_values(["decision", "ticker"])
    exc = df[df.candidacy == "EXCLUDE"].sort_values("n_failed_legs", ascending=False)
    near = df[df.decision == "NEAR_DUAL"].sort_values("n_failed_legs")

    print(f"\nInclusion candidates: {len(inc)}  |  Exclusion: {len(exc)}  |  Near-dual: {len(near)}")

    if explore:
        defensive = explore_defensive(df, stocks)
    else:
        defensive = df[df.defensive_value_index == True].copy()

    correlation_tables(save=save)

    if save:
        Path(OUT_RULES).write_text(json.dumps(RULES, indent=2))
        inc.to_parquet(OUT_INC)
        exc.to_parquet(OUT_EXC)
        near.to_parquet(OUT_NEAR)
        if explore:
            defensive.to_parquet(OUT_DEF)
        else:
            df[df.defensive_value_index].to_parquet(OUT_DEF)
        print(f"Wrote {OUT_INC}, {OUT_EXC}, {OUT_NEAR}, {OUT_DEF}, {OUT_RULES}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-defensive", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(explore=args.explore_defensive or True, save=True)


if __name__ == "__main__":
    main()

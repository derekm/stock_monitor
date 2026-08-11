#!/usr/bin/env python3
"""
implied_r_screen.py — Ohlson-Rueangsuwan (2026) implied cost-of-capital screen.

"Formal Equity Valuation: Overview and Limits" (SSRN 6280638) argues the most
defensible use of valuation formulas is NOT estimating P (fragile r and g) but
INFERRING r from the current price and fundamentals, then asking: "does the
market's implied r look high (cheap) or low (expensive)?"

Reduced-form RIV (g = r/2), Ohlson & Rueangsuwan eq:
    P = -BV + 2*X(1)/r          ->   r = 2*X(1)/(P + BV)

With X(1) = ROE*BV (expected next-period earnings on current book) and
BV = P/(P/B):

    r_implied = 2*ROE/(P/B + 1)

That's the whole screen: one observable per ticker, no analyst forecasts, no g.

Benchmarks (from the paper):
  - forward P/E = 1/r
  - sanity triplet: any two of {P/B < 1, r > 1/forward-P/E, r > ROE} imply the third
  - r > ROE means the market demands more than the firm earns on book -> cheap

Usage:
  python implied_r_screen.py            # print the screen
  python implied_r_screen.py --save     # write implied_r_screen.csv/.parquet
  python implied_r_screen.py --min-cap 10   # only names > $10B market cap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_PQ = DATA_DIR / "implied_r_screen.parquet"

# Fair-value thresholds (paper's worked example uses r = 9% for a risky firm;
# 8% is the textbook midpoint). CHEAP = market demands > 12%, EXPENSIVE < 6%.
R_CHEAP = 0.12
R_FAIR_HI = 0.10
R_FAIR_LO = 0.07
R_EXPENSIVE = 0.06

# Fair-value range endpoints: price implied by the full RIV reduced form
# P = -BV + 2*EPS1/r at a 7% vs 10% required return. r=10% -> lower price
# (conservative bound), r=7% -> higher price (generous bound). A stock inside
# the range is fairly valued; below the low end is undervalued vs fair value;
# above the high end is overvalued even at a cheap required return.
FV_R_LO = 0.10   # conservative required return -> fair-value LOW bound
FV_R_HI = 0.07   # generous required return    -> fair-value HIGH bound
FV_R_MID = 0.085  # midpoint for the point estimate


def fair_value_range(roe, bvps):
    """Full RIV reduced-form price P = -BV + 2*EPS1/r at r in {7, 8.5, 10}%.

    EPS1 = ROE * BV (next-period earnings on current book). Returns the three
    fair values plus low/high bounds of the 7-10% band. None-safe.
    """
    eps1 = roe * bvps
    fv_lo = -bvps + 2.0 * eps1 / FV_R_LO if eps1 else np.nan
    fv_mid = -bvps + 2.0 * eps1 / FV_R_MID if eps1 else np.nan
    fv_hi = -bvps + 2.0 * eps1 / FV_R_HI if eps1 else np.nan
    return fv_lo, fv_mid, fv_hi


def latest_price() -> pd.Series:
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    p = p.sort_values("date").groupby("ticker").tail(1)
    return p.set_index("ticker")["close"]


def latest_market_cap_b() -> pd.Series:
    """Latest daily market cap in $B (fresh — beats the quarterly fundamentals snapshot).

    Falls back to fundamentals.market_cap_b where the daily column is missing
    (ETFs/ADRs without fundamentals-derived shares).
    """
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "market_cap"])
    p = p[p["market_cap"].notna()].sort_values("date").groupby("ticker").tail(1)
    mc = p.set_index("ticker")["market_cap"] / 1e9
    f = pd.read_parquet(FUND)
    f = f.sort_values("as_of_date").groupby("ticker").tail(1)
    fb = f.set_index("ticker")["market_cap_b"]
    return mc.combine_first(fb)


def latest_fundamentals() -> pd.DataFrame:
    f = pd.read_parquet(FUND)
    f = f.sort_values("as_of_date").groupby("ticker").tail(1)
    return f.set_index("ticker")


def screen(min_cap_b: float = 0.0) -> pd.DataFrame:
    px = latest_price()
    f = latest_fundamentals()
    df = pd.DataFrame({"price": px})
    df["roe"] = f["roe"]
    df["pb_ratio"] = f["pb_ratio"]
    df["mktcap_b"] = latest_market_cap_b().reindex(df.index)
    df["ev_ebitda"] = f["ev_ebitda"]
    df["roic"] = f["roic"]
    df["as_of"] = f["as_of_date"]

    df = df.dropna(subset=["price", "roe", "pb_ratio"])
    df = df[df["price"] > 0]
    if min_cap_b:
        df = df[df["mktcap_b"] >= min_cap_b]

    # RIV reduced form, g = r/2:  r = 2*ROE/(P/B + 1)
    df["implied_r"] = 2.0 * df["roe"] / (df["pb_ratio"] + 1.0)
    # forward P/E benchmark = 1/r
    df["fwd_pe_bench"] = 1.0 / df["implied_r"].replace(0, np.nan)
    # book per share (for the full RIV expression, informational)
    df["bvps"] = df["price"] / df["pb_ratio"]

    # Triplet sanity: any two of {P/B<1, r>1/fwdPE, r>ROE} imply the third.
    # 1/fwdPE == implied_r by construction, so check r vs ROE and P/B vs 1.
    df["r_gt_roe"] = df["implied_r"] > df["roe"]
    df["pb_lt_1"] = df["pb_ratio"] < 1.0
    df["triplet_ok"] = (df["r_gt_roe"] & df["pb_lt_1"]) | (
        (df["r_gt_roe"] != df["pb_lt_1"]) & (df["implied_r"] > df["roe"] * df["pb_ratio"] / (1 + df["pb_ratio"]))
    )

    # Value verdict vs paper thresholds
    def verdict(r):
        if r >= R_CHEAP:
            return "CHEAP"
        if r > R_FAIR_HI:
            return "Fair-ish"
        if r >= R_FAIR_LO:
            return "FAIR"
        if r > R_EXPENSIVE:
            return "Rich"
        return "EXPENSIVE"

    df["verdict"] = df["implied_r"].apply(verdict)
    df["implied_r_pct"] = (df["implied_r"] * 100).round(1)
    df["fwd_pe_bench"] = df["fwd_pe_bench"].round(1)
    df["price"] = df["price"].round(2)
    df["bvps"] = df["bvps"].round(2)

    # Fair-value range at r = 7% / 8.5% / 10% (full RIV reduced form)
    fv = df.apply(lambda r: pd.Series(fair_value_range(r["roe"], r["bvps"])), axis=1)
    df["fv_lo_r10"] = fv[0].round(2)
    df["fv_mid_r8p5"] = fv[1].round(2)
    df["fv_hi_r7"] = fv[2].round(2)

    # Where is price vs the 7-10% fair band?
    def vs_fair(row):
        price, lo, hi = row["price"], row["fv_lo_r10"], row["fv_hi_r7"]
        if np.isnan(lo) or np.isnan(hi):
            return None
        if price < lo:
            return "BELOW_FAIR"
        if price > hi:
            return "ABOVE_FAIR"
        return "IN_FAIR"

    df["vs_fair"] = df.apply(vs_fair, axis=1)
    # % gap between price and the mid (r=8.5%) fair value
    df["fv_gap_pct"] = ((df["price"] / df["fv_mid_r8p5"] - 1.0) * 100).round(1)

    df = df.reset_index().rename(columns={"index": "ticker"})
    df = df.sort_values("implied_r", ascending=False)
    cols = ["ticker", "price", "bvps", "pb_ratio", "roe", "implied_r_pct",
            "fwd_pe_bench", "verdict", "mktcap_b", "ev_ebitda", "roic",
            "r_gt_roe", "pb_lt_1", "triplet_ok", "as_of",
            "fv_lo_r10", "fv_mid_r8p5", "fv_hi_r7", "vs_fair", "fv_gap_pct"]
    return df[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--min-cap", type=float, default=0.0, help="min market cap $B")
    ap.add_argument("--top", type=int, default=25, help="rows to print per verdict")
    args = ap.parse_args()

    df = screen(min_cap_b=args.min_cap)
    if df.empty:
        print("no tickers passed the filter")
        return

    print(f"=== Implied cost-of-capital screen ({len(df)} tickers) ===")
    print("Formula: r = 2*ROE/(P/B + 1)  [RIV reduced form, g=r/2, Ohlson & Rueangsuwan 2026]")
    print(f"Thresholds: CHEAP r>=12% | Fair 7-10% | EXPENSIVE r<=6%")
    print("Fair-value band: P = -BV + 2*EPS1/r at r = 7%/8.5%/10% (full RIV reduced form)\n")
    for v in ["CHEAP", "Fair-ish", "FAIR", "Rich", "EXPENSIVE"]:
        sub = df[df["verdict"] == v]
        if sub.empty:
            continue
        print(f"--- {v} ({len(sub)}) ---")
        show_cols = ["ticker", "price", "pb_ratio", "roe", "implied_r_pct",
                     "fv_lo_r10", "fv_mid_r8p5", "fv_hi_r7", "vs_fair", "fv_gap_pct"]
        print(sub[show_cols].head(args.top).to_string(index=False))
        print()

    # market stats
    med = df["implied_r_pct"].median()
    print(f"Median implied r: {med:.1f}% | n={len(df)}")
    print(f"  CHEAP count: {(df['verdict']=='CHEAP').sum()} | "
          f"EXPENSIVE count: {(df['verdict']=='EXPENSIVE').sum()}")
    vb = df["vs_fair"].value_counts(dropna=False)
    print(f"  vs fair band: {dict(vb)}")

    if args.save:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_PQ)
        print(f"\nWrote {OUT_PQ} ({len(df)} rows)")


if __name__ == "__main__":
    main()

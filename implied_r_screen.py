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

# ── Financial-distortion guard (Damodaran) ─────────────────────────────
# For banks/insurers/REITs/utilities, book value ≈ invested assets and ROE is
# levered by float/deposits. The RIV reduced form r = 2*ROE/(P/B+1) therefore
# MECHANICALLY overstates cheapness (inflated ROE + depressed P/B). These
# sectors' implied-r is unreliable as a standalone value signal.
DISTORTED_SECTORS = {"Financials", "Utilities", "Real Estate", "Financial", "Multi-Sector"}

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


def load_wacc_per_ticker() -> pd.Series:
    """Load WACC per ticker from Damodaran computation, or empty."""
    path = DATA_DIR / "wacc_per_ticker.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if "ticker" in df.columns and "wacc" in df.columns:
        return df.set_index("ticker")["wacc"]
    return pd.Series(dtype=float)


def load_cost_of_equity_per_ticker() -> pd.Series:
    """Load cost of equity per ticker from Damodaran computation, or empty."""
    path = DATA_DIR / "wacc_per_ticker.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if "ticker" in df.columns and "cost_of_equity" in df.columns:
        return df.set_index("ticker")["cost_of_equity"]
    return pd.Series(dtype=float)


def price_series() -> pd.DataFrame:
    """Full daily close matrix (wide) for beta / risk computation."""
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()


def _beta_map(wide: pd.DataFrame, tickers) -> pd.Series:
    """1y weekly beta vs equal-weight market (close-based). NaN-safe."""
    if wide is None or len(wide) < 60:
        return pd.Series(dtype=float)
    r = np.log(wide / wide.shift(1)).resample("W").sum()
    mkt = r.mean(axis=1)
    out = {}
    for t in tickers:
        if t not in r.columns:
            continue
        x = r[[t]].dropna().iloc[-60:]
        if len(x) < 30:
            continue
        y = mkt.reindex(x.index).dropna()
        x = x.reindex(y.index).iloc[:, 0]
        if len(x) < 30:
            continue
        cov = np.cov(x, y)
        out[t] = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
    return pd.Series(out)


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


def sector_map() -> pd.Series:
    """ticker -> GICS sector. Primary: sp500_constituents (full universe).
    Fallback: monitored_stocks. Unknown otherwise."""
    sec = {}
    sp = DATA_DIR / "sp500_constituents.parquet"
    if sp.exists():
        s = pd.read_parquet(sp, columns=["ticker", "gics_sector"])
        for _, r in s.drop_duplicates("ticker").iterrows():
            sec[str(r["ticker"]).upper()] = str(r["gics_sector"])
    if STOCKS.exists():
        s = pd.read_parquet(STOCKS, columns=["ticker", "sector"])
        for _, r in s.drop_duplicates("ticker").iterrows():
            sec.setdefault(str(r["ticker"]).upper(), str(r["sector"]))
    out = pd.Series(sec, dtype=str)
    out.index.name = "ticker"
    return out


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
    wide = price_series()

    df = df.dropna(subset=["price", "roe", "pb_ratio"])
    df = df[df["price"] > 0]
    if min_cap_b:
        df = df[df["mktcap_b"] >= min_cap_b]

    # Sector + financial-distortion guard
    sec = sector_map().reindex(df.index)
    df["sector"] = sec.fillna("Unknown")
    df["is_financial"] = df["sector"].isin(DISTORTED_SECTORS)
    # For financials, the RIV implied-r is structurally inflated. Flag it
    # (distortion_flag=True) and keep the raw value but mark unreliable.
    df["r_distorted"] = df["is_financial"]

    # ── Damodaran excess-return metric (financials' correct value driver) ──
    # Value = BV + PV(excess returns), excess return = ROE − cost of equity.
    # COE via CAPM = rf + levered-beta·ERP. Levered beta from the price beta
    # (which already embeds leverage) + a leverage penalty for high D/E, since
    # Damodaran stresses financials' equity-only risk rises with leverage and
    # deposits/float are raw material, not capital. ERP 4.23%, rf 4.18%.
    RF = 0.0418
    ERP = 0.0423
    wacc_series = load_wacc_per_ticker()
    coe_series = load_cost_of_equity_per_ticker()
    betas = _beta_map(wide, df.index)
    df["beta"] = betas.reindex(df.index).fillna(1.0)
    # Leverage penalty: D/E above ~2x raises COE (beta-like). Use a gentle
    # additive term so high-D/E financials aren't read as free value.
    de = pd.to_numeric(f.get("debt_to_equity"), errors="coerce").reindex(df.index)
    df["debt_to_equity"] = de
    lev_prem = np.clip((de - 2.0) / 5.0, 0.0, 0.05).fillna(0.0)
    df["cost_of_equity"] = RF + df["beta"] * ERP + lev_prem
    df["excess_return"] = df["roe"] - df["cost_of_equity"]
    df["excess_ret_pct"] = (df["excess_return"] * 100).round(1)
    # Operative value verdict for financials: value created iff ROE > COE.
    def excess_verdict(row):
        if row["excess_return"] >= 0.03:
            return "CREATES_VALUE"
        if row["excess_return"] >= 0.0:
            return "AT_COST"
        return "DESTROYS_VALUE"
    df["excess_ret_verdict"] = df.apply(excess_verdict, axis=1)

    # ── Damodaran per-ticker WACC and cost of equity ──────────────────────
    # Use Damodaran-computed WACC per ticker (from wacc_per_ticker.parquet)
    # which uses sector betas, CRP, and synthetic ratings from interest coverage.
    df["wacc_damodaran"] = wacc_series.reindex(df.index)
    df["cost_of_equity_damodaran"] = coe_series.reindex(df.index)
    # Implied r using Damodaran WACC as the required return benchmark
    # r_implied_damodaran = 2*ROE/(P/B + 1) but with WACC as floor
    df["implied_r_damodaran"] = df["implied_r"]
    # For tickers with Damodaran WACC, use it as the fair-value benchmark
    # instead of the static 7-10% band
    has_wacc = df["wacc_damodaran"].notna()
    df.loc[has_wacc, "implied_r_damodaran"] = df.loc[has_wacc, "implied_r"]
    # Excess return using Damodaran COE
    df["excess_return_damodaran"] = df["roe"] - df["cost_of_equity_damodaran"]
    df["excess_ret_damodaran_pct"] = (df["excess_return_damodaran"] * 100).round(1)

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

    # ── g-sensitivity (Ohlson: g is under-identified) ────────────────────
    # The fair-value band uses an implicit g=r/2. Report how the implied-r
    # verdict shifts under alternative growth anchors. RIV reduced form at
    # general g: r = (ROE·(1+g) + g·(P/B)) / (P/B + 1). Solving implied-r with
    # g=0, g=r/2 (paper default), g=0.75r.
    pb = df["pb_ratio"]
    roe = df["roe"]
    # g=0 (no-growth): r = ROE·(1)/(PB+1) ... use r = ROE/PB is wrong; use the
    # clean-surplus PVED growth-adjusted form: r_g0 = ROE / PB  (E/P + implied)
    # Simplest defensible anchors: report ROE/PB (g=0 perpetual) and ROE (g=r,
    # the ceiling). These bracket the paper's r/2.
    df["r_g0"] = roe / pb          # implied r under no growth (E/P-style)
    df["r_g_ceiling"] = roe        # implied r ceiling (g→r)
    df["r_g0_pct"] = (df["r_g0"] * 100).round(1)
    df["r_g_ceiling_pct"] = (df["r_g_ceiling"] * 100).round(1)
    # verdict under the two anchors (distorted financials excluded from clean)
    df["cheap_robust"] = (df["r_g0"] >= R_CHEAP) & ~df["r_distorted"]
    df["rich_robust"] = (df["r_g_ceiling"] < R_EXPENSIVE) & ~df["r_distorted"]

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
    # Clean implied-r: drop the distorted financial value so the CHEAP/FAIR
    # screen isn't polluted by book-heavy sectors' mechanically-high r.
    df["implied_r_clean"] = df["implied_r"].where(~df["r_distorted"])
    df["implied_r_clean_pct"] = (df["implied_r_clean"] * 100).round(1)
    cols = ["ticker", "sector", "is_financial", "r_distorted", "price", "bvps",
            "pb_ratio", "roe", "beta", "debt_to_equity", "implied_r_pct", "implied_r_clean_pct",
            "cost_of_equity", "cost_of_equity_damodaran", "wacc_damodaran",
            "excess_ret_pct", "excess_ret_damodaran_pct", "excess_ret_verdict",
            "r_g0_pct", "r_g_ceiling_pct", "cheap_robust", "rich_robust",
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
    print("Fair-value band: P = -BV + 2*EPS1/r at r = 7%/8.5%/10% (full RIV reduced form)")
    print(f"NOTE: {int(df['r_distorted'].sum())} financial/utility/REIT names flagged (r_distorted=True) — "
          f"their implied-r is unreliable (book-heavy, see clean col).\n")
    # Damodaran WACC summary
    n_wacc = df["wacc_damodaran"].notna().sum()
    if n_wacc > 0:
        print(f"\nDamodaran WACC coverage: {n_wacc}/{len(df)} tickers")
        print(f"  Median WACC: {df['wacc_damodaran'].median():.2%}")
        print(f"  Median COE (Damodaran): {df['cost_of_equity_damodaran'].median():.2%}")
        print(f"  Median Excess Return (Damodaran): {df['excess_ret_damodaran_pct'].median():.1f}%")

    for v in ["CHEAP", "Fair-ish", "FAIR", "Rich", "EXPENSIVE"]:
        sub = df[df["verdict"] == v]
        if sub.empty:
            continue
        print(f"--- {v} ({len(sub)}) ---")
        show_cols = ["ticker", "sector", "r_distorted", "price", "pb_ratio", "roe",
                     "implied_r_clean_pct", "fv_lo_r10", "fv_mid_r8p5", "fv_hi_r7",
                     "vs_fair", "fv_gap_pct"]
        print(sub[show_cols].head(args.top).to_string(index=False))
        print()

    # market stats
    med = df["implied_r_clean_pct"].dropna().median()
    print(f"Median implied r (clean, ex-financials): {med:.1f}% | n={df['implied_r_clean_pct'].notna().sum()}")
    print(f"  CHEAP count (clean): {(df['implied_r_clean_pct'] >= R_CHEAP*100).sum()} | "
          f"EXPENSIVE count (clean): {(df['implied_r_clean_pct'] < R_EXPENSIVE*100).sum()}")
    vb = df["vs_fair"].value_counts(dropna=False)
    print(f"  vs fair band: {dict(vb)}")

    if args.save:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT_PQ)
        print(f"\nWrote {OUT_PQ} ({len(df)} rows)")


if __name__ == "__main__":
    main()

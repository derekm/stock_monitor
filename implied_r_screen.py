#!/usr/bin/env python3
"""
implied_r_screen.py — Ohlson-Rueangsuwan (2026) implied cost-of-capital screen
with dynamic ERP options.

Three ERP sources:
  1. damodaran — Damodaran implied ERP (annual/semi-annual, from erp_history.parquet)
  2. interpolated — Monthly or daily interpolation of Damodaran ERP
  3. spy_sma — price-to-200dma heuristic (NOT Shiller CAPE; labeled honestly)

Usage:
  python implied_r_screen.py --save
  python implied_r_screen.py --save --erp damodaran --erp-freq monthly
  python implied_r_screen.py --save --erp shiller
  python implied_r_screen.py --save --erp interpolated --erp-freq daily
  python implied_r_screen.py --compare  # compare ERP sources
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import requests
from scipy.interpolate import interp1d

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_PQ = DATA_DIR / "implied_r_screen.parquet"

# Fair-value thresholds (paper)
R_CHEAP = 0.12
R_FAIR_HI = 0.10
R_FAIR_LO = 0.07
R_EXPENSIVE = 0.06

FV_R_LO = 0.10
FV_R_HI = 0.07
FV_R_MID = 0.085


# ─────────────────────────────────────────────────────────────────────────────
# ERP SOURCES
# ─────────────────────────────────────────────────────────────────────────────

def load_damodaran_erp(freq: str = "semi_annual") -> pd.DataFrame:
    """Load Damodaran implied ERP from erp_history.parquet.
    
    freq: 'annual', 'semi_annual', 'monthly', 'daily'
      - annual: yearly average
      - semi_annual: Jan/Jul points (native granularity)
      - monthly: interpolated to monthly
      - daily: forward-filled to daily
    
    Returns DataFrame with columns: [date, erp, source]
    """
    path = DATA_DIR / "erp_history.parquet"
    if not path.exists():
        print(f"WARNING: {path} not found — using default ERP=4.23%")
        return pd.DataFrame({"date": [pd.Timestamp.now()], "erp": [0.0423], "source": ["default"]})
    
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["implied_erp"])
    df = df.sort_values("date").reset_index(drop=True)
    
    if freq == "semi_annual":
        return df.rename(columns={"implied_erp": "erp"})[["date", "erp", "source"]]
    
    if freq == "annual":
        df["year"] = df["date"].dt.year
        annual = df.groupby("year")["implied_erp"].mean().reset_index()
        annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-06-30")
        annual["source"] = "damodaran_annual_avg"
        return annual.rename(columns={"implied_erp": "erp"})[["date", "erp", "source"]]
    
    # For monthly or daily interpolation
    min_date = df["date"].min()
    max_date = df["date"].max()
    
    if freq == "monthly":
        target_dates = pd.date_range(min_date, max_date, freq="MS")
    else:  # daily
        target_dates = pd.date_range(min_date, max_date, freq="D")
    
    # Interpolate
    known_dates = df["date"].map(pd.Timestamp.toordinal).values
    known_erps = df["implied_erp"].values
    target_ordinals = target_dates.map(pd.Timestamp.toordinal).values
    
    interp = interp1d(known_dates, known_erps, kind="linear", fill_value="extrapolate")
    interp_erps = np.clip(interp(target_ordinals), 0.01, 0.15)
    
    result = pd.DataFrame({
        "date": target_dates,
        "erp": interp_erps,
        "source": f"damodaran_interp_{freq}",
    })
    return result


def load_shiller_erp() -> pd.DataFrame:
    """Build Shiller-style earnings-yield ERP proxy.
    
    Proxy: ERP = (1 / CAPE) − rf
    Where CAPE ≈ 200dma price / average earnings yield (we proxy with price/200dma).
    
    We map the price-to-200dma ratio to an ERP using historical correlation:
    - When price >> 200dma (expensive), ERP is low
    - When price << 200dma (cheap), ERP is high
    
    Baseline: at price/200dma = 1.0, ERP ≈ 4.5% (long-term average).
    For every 10% increase in price/200dma, ERP drops ~0.3%.
    """
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    
    # Get S&P 500 prices (SPY as proxy)
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").copy()
    if spy.empty:
        spy = prices.groupby("date")["close"].mean().reset_index().sort_values("date")
    
    spy["date"] = pd.to_datetime(spy["date"])
    
    # Compute 200-day moving average
    spy["sma_200"] = spy["close"].rolling(200, min_periods=100).mean()
    spy["price_to_sma200"] = spy["close"] / spy["sma_200"]
    
    # Map price_to_sma200 to ERP
    # At ratio 1.0 → ERP ~ 4.5% (long-term average)
    # At ratio 1.1 → ERP ~ 3.5% (slightly expensive)
    # At ratio 0.8 → ERP ~ 6.5% (cheap)
    spy["erp"] = 0.045 - (spy["price_to_sma200"] - 1.0) * 0.10
    spy["erp"] = spy["erp"].clip(0.02, 0.10)
    spy["source"] = "spy_sma_heuristic"
    return spy[["date", "erp", "source"]].dropna(subset=["erp"])


def load_cape_erp() -> pd.DataFrame:
    """Shiller CAPE ERP = 1/PE10 - long rate. Uses datahub s-and-p-500 series."""
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
        "https://datahub.io/core/s-and-p-500/r/data.csv",
    ]
    UA = {"User-Agent": "personal-research derek.moore@example.com"}
    raw = None
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and "PE10" in r.text[:2000] or "PE10" in r.text:
                raw = pd.read_csv(io.StringIO(r.text))
                break
        except Exception:
            continue
    if raw is None or raw.empty:
        print("WARNING: CAPE series unavailable — falling back to Damodaran ERP")
        return load_damodaran_erp("semi_annual")
    raw.columns = [c.strip() for c in raw.columns]
    pe = "PE10" if "PE10" in raw.columns else [c for c in raw.columns if "PE" in c.upper()][0]
    raw["date"] = pd.to_datetime(raw["Date"] if "Date" in raw.columns else raw.iloc[:, 0], errors="coerce")
    raw["cape"] = pd.to_numeric(raw[pe], errors="coerce")
    rf_col = next((c for c in raw.columns if "Long" in c or "Interest" in c), None)
    rf = pd.to_numeric(raw[rf_col], errors="coerce") / 100.0 if rf_col else 0.0418
    out = raw.dropna(subset=["date", "cape"]).copy()
    out["erp"] = (1.0 / out["cape"] - rf).clip(0.01, 0.15)
    out["source"] = "shiller_cape"
    return out[["date", "erp", "source"]]


def load_erp(erp_source: str = "damodaran", erp_freq: str = "semi_annual") -> pd.DataFrame:
    """Load ERP from the specified source.
    
    Args:
        erp_source: 'damodaran', 'interpolated', 'spy_sma', 'cape' (shiller→cape)
        erp_freq: 'annual', 'semi_annual', 'monthly', 'daily'
    
    Returns DataFrame with columns: [date, erp, source]
    """
    if erp_source == "damodaran":
        return load_damodaran_erp(freq=erp_freq if erp_freq != "daily" else "semi_annual")
    elif erp_source == "interpolated":
        return load_damodaran_erp(freq=erp_freq)
    elif erp_source in ("cape", "shiller"):
        return load_cape_erp()
    elif erp_source == "spy_sma":
        return load_shiller_erp()
    else:
        raise ValueError(f"Unknown ERP source: {erp_source}")


def get_erp_for_date(erp_table: pd.DataFrame, target_date, method: str = "nearest") -> float:
    """Get ERP value for a specific date from the ERP table."""
    if erp_table.empty:
        return 0.0423
    
    target_date = pd.Timestamp(target_date)
    
    if method == "nearest":
        idx = erp_table["date"].searchsorted(target_date)
        if idx == 0:
            return erp_table.iloc[0]["erp"]
        if idx >= len(erp_table):
            return erp_table.iloc[-1]["erp"]
        before = erp_table.iloc[idx - 1]
        after = erp_table.iloc[idx]
        if abs((target_date - before["date"]).days) <= abs((target_date - after["date"]).days):
            return before["erp"]
        return after["erp"]
    else:  # ffill
        mask = erp_table["date"] <= target_date
        if not mask.any():
            return erp_table.iloc[0]["erp"]
        return erp_table[mask].iloc[-1]["erp"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DISTORTED_SECTORS = {"Financials", "Utilities", "Real Estate", "Financial", "Multi-Sector"}


def fair_value_range(roe, bvps):
    """Full RIV reduced-form price P = -BV + 2*EPS1/r at r in {7, 8.5, 10}%."""
    eps1 = roe * bvps
    if not eps1:
        return np.nan, np.nan, np.nan
    fv_lo = -bvps + 2.0 * eps1 / FV_R_LO
    fv_mid = -bvps + 2.0 * eps1 / FV_R_MID
    fv_hi = -bvps + 2.0 * eps1 / FV_R_HI
    return fv_lo, fv_mid, fv_hi


def latest_price() -> pd.Series:
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    p = p.sort_values("date").groupby("ticker").tail(1)
    return p.set_index("ticker")["close"]


def load_wacc_per_ticker() -> pd.Series:
    path = DATA_DIR / "wacc_per_ticker.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if "ticker" in df.columns and "wacc" in df.columns:
        return df.set_index("ticker")["wacc"]
    return pd.Series(dtype=float)


def load_cost_of_equity_per_ticker() -> pd.Series:
    path = DATA_DIR / "wacc_per_ticker.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if "ticker" in df.columns and "cost_of_equity" in df.columns:
        return df.set_index("ticker")["cost_of_equity"]
    return pd.Series(dtype=float)


def price_series() -> pd.DataFrame:
    p = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()


def _beta_map(wide: pd.DataFrame, tickers) -> pd.Series:
    """1y weekly beta vs equal-weight market (close-based)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def screen(min_cap_b: float = 0.0, erp_source: str = "damodaran", erp_freq: str = "semi_annual") -> pd.DataFrame:
    # Load ERP table
    erp_table = load_erp(erp_source=erp_source, erp_freq=erp_freq)
    
    # Get current ERP
    current_erp = get_erp_for_date(erp_table, pd.Timestamp.now(), method="ffill")
    print(f"ERP source: {erp_source} | freq: {erp_freq} | current ERP: {current_erp:.2%}")
    
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

    sec = sector_map().reindex(df.index)
    df["sector"] = sec.fillna("Unknown")
    df["is_financial"] = df["sector"].isin(DISTORTED_SECTORS)
    df["r_distorted"] = df["is_financial"]

    # ── Cost of equity with dynamic ERP ──
    RF = 0.0418
    ERP = current_erp
    
    wacc_series = load_wacc_per_ticker()
    coe_series = load_cost_of_equity_per_ticker()
    betas = _beta_map(wide, df.index)
    df["beta"] = betas.reindex(df.index).fillna(1.0)
    
    de = pd.to_numeric(f.get("debt_to_equity"), errors="coerce").reindex(df.index)
    df["debt_to_equity"] = de
    lev_prem = np.clip((de - 2.0) / 5.0, 0.0, 0.05).fillna(0.0)
    df["cost_of_equity"] = RF + df["beta"] * ERP + lev_prem
    df["excess_return"] = df["roe"] - df["cost_of_equity"]
    df["excess_ret_pct"] = (df["excess_return"] * 100).round(1)
    
    def excess_verdict(row):
        if row["excess_return"] >= 0.03:
            return "CREATES_VALUE"
        if row["excess_return"] >= 0.0:
            return "AT_COST"
        return "DESTROYS_VALUE"
    
    df["excess_ret_verdict"] = df.apply(excess_verdict, axis=1)
    
    # Damodaran per-ticker WACC
    df["wacc_damodaran"] = wacc_series.reindex(df.index)
    df["cost_of_equity_damodaran"] = coe_series.reindex(df.index)
    df["implied_r_damodaran"] = 2.0 * df["roe"] / (df["pb_ratio"] + 1.0)
    df["excess_return_damodaran"] = df["roe"] - df["cost_of_equity_damodaran"]
    df["excess_ret_damodaran_pct"] = (df["excess_return_damodaran"] * 100).round(1)

    # RIV reduced form: r = 2*ROE/(P/B + 1)
    df["implied_r"] = 2.0 * df["roe"] / (df["pb_ratio"] + 1.0)
    df["fwd_pe_bench"] = 1.0 / df["implied_r"].replace(0, np.nan)
    df["bvps"] = df["price"] / df["pb_ratio"]

    # Triplet sanity
    df["r_gt_roe"] = df["implied_r"] > df["roe"]
    df["pb_lt_1"] = df["pb_ratio"] < 1.0
    df["triplet_ok"] = (df["r_gt_roe"] & df["pb_lt_1"]) | (
        (df["r_gt_roe"] != df["pb_lt_1"]) & (df["implied_r"] > df["roe"] * df["pb_ratio"] / (1 + df["pb_ratio"]))
    )

    # Value verdict
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

    # Fair-value range
    fv = df.apply(lambda r: pd.Series(fair_value_range(r["roe"], r["bvps"])), axis=1)
    df["fv_lo_r10"] = fv[0].round(2)
    df["fv_mid_r8p5"] = fv[1].round(2)
    df["fv_hi_r7"] = fv[2].round(2)

    # g-sensitivity
    pb = df["pb_ratio"]
    roe = df["roe"]
    df["r_g0"] = roe / pb
    df["r_g_ceiling"] = roe
    df["r_g0_pct"] = (df["r_g0"] * 100).round(1)
    df["r_g_ceiling_pct"] = (df["r_g_ceiling"] * 100).round(1)
    df["cheap_robust"] = (df["r_g0"] >= R_CHEAP) & ~df["r_distorted"]
    df["rich_robust"] = (df["r_g_ceiling"] < R_EXPENSIVE) & ~df["r_distorted"]

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
    df["fv_gap_pct"] = ((df["price"] / df["fv_mid_r8p5"] - 1.0) * 100).round(1)

    df = df.reset_index().rename(columns={"index": "ticker"})
    df = df.sort_values("implied_r", ascending=False)
    df["implied_r_clean"] = df["implied_r"].where(~df["r_distorted"])
    df["implied_r_clean_pct"] = (df["implied_r_clean"] * 100).round(1)
    
    # Add ERP metadata
    df["erp_used"] = erp_source
    df["erp_value"] = current_erp
    
    cols = ["ticker", "sector", "is_financial", "r_distorted", "price", "bvps",
            "pb_ratio", "roe", "beta", "debt_to_equity", "implied_r_pct", "implied_r_clean_pct",
            "cost_of_equity", "cost_of_equity_damodaran", "wacc_damodaran",
            "excess_ret_pct", "excess_ret_damodaran_pct", "excess_ret_verdict",
            "r_g0_pct", "r_g_ceiling_pct", "cheap_robust", "rich_robust",
            "fwd_pe_bench", "verdict", "mktcap_b", "ev_ebitda", "roic",
            "r_gt_roe", "pb_lt_1", "triplet_ok", "as_of",
            "fv_lo_r10", "fv_mid_r8p5", "fv_hi_r7", "vs_fair", "fv_gap_pct",
            "erp_used", "erp_value"]
    return df[cols]


def compare_erp_sources() -> pd.DataFrame:
    """Compare ERP from different sources for validation."""
    sources = {
        "damodaran_semi": ("damodaran", "semi_annual"),
        "damodaran_monthly": ("interpolated", "monthly"),
        "damodaran_daily": ("interpolated", "daily"),
        "spy_sma": ("spy_sma", "daily"),
        "cape": ("cape", "monthly"),
    }
    
    results = []
    for name, (source, freq) in sources.items():
        try:
            erp_table = load_erp(erp_source=source, erp_freq=freq)
            current_erp = get_erp_for_date(erp_table, pd.Timestamp.now(), method="ffill")
            results.append({
                "source": name,
                "current_erp": current_erp,
                "n_points": len(erp_table),
                "date_range": f"{erp_table['date'].min()} to {erp_table['date'].max()}",
            })
        except Exception as e:
            results.append({
                "source": name,
                "current_erp": np.nan,
                "n_points": 0,
                "date_range": str(e),
            })
    
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--min-cap", type=float, default=0.0, help="min market cap $B")
    ap.add_argument("--top", type=int, default=25, help="rows to print per verdict")
    ap.add_argument("--erp", default="damodaran", 
                    choices=["damodaran", "interpolated", "spy_sma", "cape", "shiller"],
                    help="ERP source")
    ap.add_argument("--erp-freq", default="semi_annual",
                    choices=["annual", "semi_annual", "monthly", "daily"],
                    help="ERP frequency (for damodaran/interpolated sources)")
    ap.add_argument("--compare", action="store_true",
                    help="Compare ERP sources and exit")
    args = ap.parse_args()

    if args.compare:
        print("=== ERP Source Comparison ===")
        comp = compare_erp_sources()
        print(comp.to_string(index=False))
        return

    df = screen(min_cap_b=args.min_cap, erp_source=args.erp, erp_freq=args.erp_freq)
    if df.empty:
        print("no tickers passed the filter")
        return

    print(f"=== Implied cost-of-capital screen ({len(df)} tickers) ===")
    print(f"ERP source: {args.erp} | freq: {args.erp_freq}")
    print(f"Formula: r = 2*ROE/(P/B + 1)  [RIV reduced form, g=r/2, Ohlson & Rueangsuwan 2026]")
    print(f"Thresholds: CHEAP r>=12% | Fair 7-10% | EXPENSIVE r<=6%")
    print(f"NOTE: {int(df['r_distorted'].sum())} financial/utility/REIT names flagged (r_distorted=True)\n")
    
    n_wacc = df["wacc_damodaran"].notna().sum()
    if n_wacc > 0:
        print(f"Damodaran WACC coverage: {n_wacc}/{len(df)} tickers")
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

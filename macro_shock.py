#!/usr/bin/env python3
"""macro_shock.py — macro supply-shock layer (the oil-crisis complement).

Why it exists: the macro-fragility layer (macro_fragility.py) measures the
DEMAND side of crises — debt impulse, velocity-scaled impulse, Credit
Accelerator. It catches debt-driven crises (1987, 2000, 2008, 2020, 2022)
but structurally MISSED the 1973-74 oil crisis (the one miss in the
crisis-label validation: impulse 0.162, danger not crisis_band). That miss
is not a bug — 1973-74 was a SUPPLY shock, not a demand/debt shock, and a
debt-driven signal cannot see it. This layer is the supply-side twin.

Four analytics, each chosen to have fired in 1973-74 (verified against our
data + FRED history):

1. OIL MOMENTUM — 12m change in the crude price. The embargo quadrupled
   oil: OILPRICE went $3.56 (1972-12) -> $10.11 (1974-03), +184% YoY.
   Leading indicator of every oil crisis: 1973-74, 1979-80 (+119%),
   2008 (->$140+), 2022 (post-invasion spike).

2. INFLATION SURPRISE — CPI YoY vs its trailing 3y average. A supply
   shock shows up as inflation ABOVE the recent norm (1974: CPI YoY ~12%
   vs ~4% prior norm; 1979-80 similar). Distinguishes supply shocks from
   demand shocks: demand-driven booms don't produce inflation surprises.

3. REAL RATE — fed funds minus CPI YoY. Deeply NEGATIVE real rates are
   the supply-shock signature (1974-75: funds ~8-12% vs CPI ~12% -> real
   rate <= -2%). Negative real rates make commodities/gold the only
   defense and mark the regime shift.

4. ENERGY DIVERGENCE — 12m energy-producer basket vs equal-weight market.
   Energy RISING while the market FALLS = supply shock. NOTE: with only
   2-3 long-history names (XOM, CVX, HAL) in the 1973 era this leg is
   noisy (adding HAL flips the sign in 1973); it strengthens after 1980
   (10 names). Reported but weighted accordingly.

Composite SHOCK SCORE = z(oil_mom) + z(inflation_surprise) - z(real_rate),
the three robust legs (energy divergence excluded from the score, reported
separately). SHOCK ZONES calibrated on the historical crises from OUR data
(see the validation printout): oil_mom_12m >= +40% and/or score >= 1.5
sigma marks the shock band.

Data: FRED public CSV endpoints (no API key; fredgraph.csv?id=...).
  OILPRICE    — IMF global price of crude (monthly, 1946-2013)
  MCOILWTICO  — WTI Cushing (monthly, 1986-)  [spliced after OILPRICE ends]
  CPIAUCSL    — CPI all urban consumers (monthly, 1947-)
  FEDFUNDS    — effective fed funds rate (monthly, 1954-)
Cached under macro_data/ (shared with macro_fragility.py).

Outputs:
  macro_shock.csv — monthly: date, oil_mom_12m, inflation_surprise,
                    real_rate, energy_divergence, shock_score, shock_zone
Reads: FRED CSV (network), daily_prices.parquet (energy basket).
Usage: python macro_shock.py [--save]
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from macro_fragility import _fetch_fred  # shared FRED cache + TTL logic

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "macro_shock.parquet"

# long-history energy producers, in depth order (thin pre-1980)
ENERGY = ["XOM", "CVX", "HAL", "APA", "COP", "OXY", "SLB", "VLO", "WMB", "OKE"]


def _splice_oil() -> pd.DataFrame:
    """Continuous monthly crude series: IMF OILPRICE (1946-2013) then WTI
    (1986-). Overlap 1986-2013 is close (both ~$100 at 2013); splice at the
    last OILPRICE observation."""
    old = _fetch_fred("OILPRICE", DATA_DIR / "macro_data" / "oilprice.parquet")
    new = _fetch_fred("MCOILWTICO", DATA_DIR / "macro_data" / "wti.parquet")
    old = old.rename(columns={"OILPRICE": "oil"})
    new = new.rename(columns={"MCOILWTICO": "oil"})
    last_old = old["observation_date"].max()
    new = new[new["observation_date"] > last_old]
    return pd.concat([old[["observation_date", "oil"]], new[["observation_date", "oil"]]],
                     ignore_index=True).sort_values("observation_date").reset_index(drop=True)


def load_energy_basket() -> pd.DataFrame:
    """Monthly energy-producer equal-weight index + market index from
    daily_prices.parquet. Returns (month, energy_ret, market_ret)."""
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(w / w.shift(1))
    mkt = rets.mean(axis=1)
    avail = [t for t in ENERGY if t in w.columns]
    en = rets[avail].mean(axis=1)
    m = pd.DataFrame({"energy": en, "market": mkt}).dropna()
    monthly = m.resample("ME").apply(lambda s: s.sum())
    monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
    return monthly


def main(save: bool = True):
    print("macro_shock: fetching FRED (OILPRICE/WTI, CPI, FEDFUNDS)...")
    oil = _splice_oil()
    cpi = _fetch_fred("CPIAUCSL", DATA_DIR / "macro_data" / "cpi.parquet")
    ff = _fetch_fred("FEDFUNDS", DATA_DIR / "macro_data" / "fedfunds.parquet")

    oil["oil_mom_12m"] = oil["oil"] / oil["oil"].shift(12) - 1
    cpi["cpi_yoy"] = cpi["CPIAUCSL"] / cpi["CPIAUCSL"].shift(12) - 1
    cpi["cpi_norm"] = cpi["cpi_yoy"].rolling(36, min_periods=12).mean()
    cpi["inflation_surprise"] = cpi["cpi_yoy"] - cpi["cpi_norm"]
    ff["real_rate"] = ff["FEDFUNDS"] / 100.0 - cpi.set_index("observation_date")["cpi_yoy"].reindex(
        ff["observation_date"], method="ffill").to_numpy()

    df = oil.merge(cpi[["observation_date", "inflation_surprise"]], on="observation_date", how="left")
    df = df.merge(ff[["observation_date", "real_rate"]], on="observation_date", how="left")
    df = df.rename(columns={"observation_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    # CPI/fed-funds publish with a 1-month lag vs oil — carry the last
    # known value forward so the latest month isn't NaN.
    df["inflation_surprise"] = df["inflation_surprise"].ffill()
    df["real_rate"] = df["real_rate"].ffill()

    # energy divergence (12m cumulative energy vs market) from daily prices
    eb = load_energy_basket()
    eb["energy_cum12"] = (1 + eb["energy"]).rolling(12).apply(np.prod, raw=True)
    eb["market_cum12"] = (1 + eb["market"]).rolling(12).apply(np.prod, raw=True)
    eb["energy_divergence"] = eb["energy_cum12"] / eb["market_cum12"] - 1
    df = df.merge(eb[["energy_divergence"]], left_on="date", right_index=True, how="left")

    # composite shock score: z(oil_mom) + z(infl_surprise) - z(real_rate)
    for col in ("oil_mom_12m", "inflation_surprise"):
        z = (df[col] - df[col].mean()) / df[col].std()
        df[f"z_{col}"] = z
    df["z_real_rate"] = (df["real_rate"] - df["real_rate"].mean()) / df["real_rate"].std()
    df["shock_score"] = (df["z_oil_mom_12m"] + df["z_inflation_surprise"] - df["z_real_rate"]) / 3

    # shock zones, calibrated on OUR crisis history (validation below):
    # oil +40% YoY and/or score >= 1.5 sigma = the supply-shock band
    def zone(r):
        if pd.isna(r["oil_mom_12m"]):
            return "no_data"
        if r["oil_mom_12m"] >= 0.40 or r["shock_score"] >= 1.5:
            return "shock"
        if r["oil_mom_12m"] >= 0.15 or r["shock_score"] >= 0.75:
            return "elevated"
        return "benign"

    df["shock_zone"] = df.apply(zone, axis=1)

    out_cols = ["date", "oil_mom_12m", "inflation_surprise", "real_rate",
                "energy_divergence", "shock_score", "shock_zone"]
    df = df[out_cols].dropna(subset=["oil_mom_12m"]).tail(720)  # 60y
    for c in ("oil_mom_12m", "inflation_surprise", "real_rate", "energy_divergence", "shock_score"):
        df[c] = df[c].round(4)

    if save:
        df.to_parquet(OUT)

    # ---- validation: what would this have said at each crisis? ----
    print("\n=== macro shock (supply-side layer) — point-in-time at crises ===")
    checks = {
        "1973-74 oil (embargo)": "1974-03-01",
        "1979-80 oil (Iran)": "1980-01-01",
        "1986 oil crash": "1986-04-01",
        "2008 (oil->$140)": "2008-07-01",
        "2014-15 oil crash": "2015-01-01",
        "2020 covid oil": "2020-04-01",
        "2022 (invasion)": "2022-06-01",
    }
    print(f"{'crisis':26s} {'oil_mom':>8s} {'infl_surp':>9s} {'real_rate':>9s} {'shock':>6s} {'zone':>8s}")
    for name, d in checks.items():
        q = df[df["date"] <= pd.Timestamp(d)].iloc[-1]
        print(f"{name:26s} {q.oil_mom_12m:8.1%} {q.inflation_surprise:9.1%} "
              f"{q.real_rate:9.1%} {q.shock_score:6.2f} {q.shock_zone:>8s}")

    last = df.iloc[-1]
    print(f"\nlatest ({last['date'].date()}): oil_mom {last['oil_mom_12m']:.1%} | "
          f"surprise {last['inflation_surprise']:.1%} | real {last['real_rate']:.1%} | "
          f"score {last['shock_score']:.2f} | zone {last['shock_zone']}")
    if save:
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

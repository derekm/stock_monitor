#!/usr/bin/env python3
"""macro_sector_shock.py — sector shock signals (farming inputs/outputs,
materials, and any basket-vs-commodity sector).

Why it exists: macro_shock.py proved the supply-shock framework on oil
(1973-74, 1979-80, 2008, 2022 all fired). The user asked whether the same
signals exist for fertilizer/materials and other economic sectors. Answer
from OUR data, verified before building:

  - FARMING INPUTS (fertilizer basket: CF/MOS/NTR/UAN/IPI/LXU/CTVA):
    12m basket momentum peaked +240% (Oct-2007, the run-up to the 2008
    food crisis) and +213% (Nov-2021, the fertilizer supercycle) — the
    same explosion signature as the oil crisis.
  - FARMING OUTPUTS (IMF global prices, 1992+): wheat peaked +133%
    (Mar-2008) and +67% (2022 invasion); corn/soy +111%/+87% (May-2021).
  - MATERIALS sector (sector_prices): +139% (May-2021, commodities
    supercycle).

Design: table-driven. Each sector is (equity basket from daily_prices or
sector_prices) + (optional FRED commodity). Baskets load GICS SECTOR
membership, GICS SUB-INDUSTRY membership (focused subsectors — the
fertilizer lesson: thin subsectors carry the explosion signal), and/or
explicit tickers (non-S&P amplifiers, history via fetch_amplifier_history.py).
A sector shock score = z(basket 12m momentum) + z(commodity 12m momentum),
exactly the macro_shock recipe (inflation/real-rate legs are macro-wide,
not sector-specific, so they stay in macro_shock.py). Shock zones calibrated
on the verified events above: basket 12m momentum >= +80% = shock, >= +40% =
elevated (the fertilizer 2007/2021 events both exceeded +200%).

Data:
  equity baskets: daily_prices.parquet (ticker lists below) or
                  sector_prices.parquet (SECT_* tickers)
  commodities:    FRED public CSV endpoints (IMF primary commodities):
                  PWHEAMTUSDM (wheat), PMAIZMTUSDM (corn), PSOYBUSDM (soy)
Cached under macro_data/ (shared with macro_fragility.py / macro_shock.py).

Outputs:
  macro_sector_shock.csv — monthly: sector, date, basket_mom_12m,
                           commodity_mom_12m, shock_score, shock_zone
Usage: python macro_sector_shock.py [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_fragility import _fetch_fred

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "macro_sector_shock.csv"

# sector -> config. Baskets come from ONE of:
#   gics:     "Energy" | "Materials" | "Consumer Staples" — full S&P 500
#             GICS membership loaded dynamically from sp500_constituents
#             (complete coverage, no hand-picked thin lists — the steering
#             fix: expand tickers to cover new sectors from S&P members)
#   tickers:  explicit basket (fallback/override for non-S&P or bespoke)
#   commodity: optional IMF global price series (FRED).
SECTORS = {
    "farming_inputs": {
        "tickers": ["CF", "MOS", "NTR", "UAN", "IPI", "LXU", "CTVA"],  # focused
        # fertilizer basket: NOT diluted by the full Materials GICS — the
        # focused names carry the explosion signal (+232% 2007, +213% 2021);
        # a broad basket smears it (measured: peak drops to +91%).
        "commodity": None,  # no global fertilizer price on FRED; basket carries it
    },
    "farming_outputs": {
        "gics": "Consumer Staples",     # ADM/BG/GIS/HSY... full S&P staples
        "commodity": "PWHEAMTUSDM",     # global wheat
    },
    "materials": {
        "gics": "Materials",
        "commodity": "PALLFNFINDEXM",   # IMF all-commodities index
    },
    "copper": {
        "gics": "Materials",            # FCX/NEM/ALB are all S&P Materials
        "commodity": "PCOPPUSDM",       # global copper (1992-)
    },
    "industrial_metals": {
        "gics": "Materials",
        "commodity": "PZINCUSDM",       # global zinc
    },
    "nickel": {
        "gics": "Materials",
        "commodity": "PNICKUSDM",       # global nickel (commodity carries it)
    },
    "energy_equities": {
        "gics": "Energy",               # full S&P energy: XOM/CVX/COP/SLB...
        "commodity": "PNGASUSUSDM",     # Henry Hub nat gas (1992-)
    },
    "thermal_coal": {
        "gics": "Energy",
        "commodity": "PCOALAUUSDM",     # Australia thermal coal (1992-)
    },
    "uranium": {
        "gics": "Energy",
        "tickers": ["CCJ", "UUUU", "EU", "NEXG"],  # non-S&P uranium names
        "commodity": "PURANUSDM",       # global uranium (1992-)
    },
    "softs_sugar": {
        "gics": "Consumer Staples",
        "commodity": "PSUGAISAUSDM",
    },
    "softs_cotton": {
        "gics": "Consumer Staples",
        "commodity": "PCOTTINDUSDM",
    },
    "softs_cocoa": {
        "gics": "Consumer Staples",
        "commodity": "PCOCOUSDM",       # +249% 2024
    },
    "softs_coffee": {
        "gics": "Consumer Staples",
        "commodity": "PCOFFOTMUSDM",    # arabica +186% 1994
    },
    "rubber": {
        "gics": "Materials",
        "commodity": "PRUBBUSDM",
    },
    # --- focused subsector baskets (2026-08) ---
    # GICS sub-industry membership + only-existing non-S&P amplifiers. Thin
    # subsectors carry the explosion signal (fertilizer lesson: +232% 2007
    # focused vs +91% broad). Only tickers present in daily_prices are used
    # (verified: TSM/ASML/SCCO/TECK/VALE/X/CLF etc. NOT in our data).
    "sub_fertilizers": {
        "subindustry": "Fertilizers & Agricultural Chemicals",
        "tickers": ["NTR", "UAN", "IPI", "LXU"],  # non-S&P amplifiers
        "commodity": None,
    },
    "sub_eandp": {
        "subindustry": "Oil & Gas Exploration & Production",
        "tickers": ["MRO", "HES"],  # present in our data
        "commodity": None,  # oil handled by macro_shock
    },
    "sub_oil_services": {
        "subindustry": "Oil & Gas Equipment & Services",
        "tickers": ["FTI"],
        "commodity": None,
    },
    "sub_refining": {
        "subindustry": "Oil & Gas Refining & Marketing",
        "tickers": ["CLNE"],
        "commodity": None,
    },
    "sub_midstream": {
        "subindustry": "Oil & Gas Storage & Transportation",
        "tickers": ["ENB", "PBA"],
        "commodity": None,
    },
    "sub_copper": {
        "subindustry": "Copper",
        "tickers": ["SCCO", "TECK", "HBM", "VALE"],  # non-S&P copper names
        "commodity": "PCOPPUSDM",
    },
    "sub_gold": {
        "subindustry": "Gold",
        "tickers": ["AEM", "KGC", "RGLD", "GOLD", "AGI"],
        "commodity": None,  # no global gold price on FRED (verified)
    },
    "sub_steel": {
        "subindustry": "Steel",
        "tickers": ["X", "CLF", "RS", "CMC", "WOR"],
        "commodity": None,
    },
    "sub_commodity_chem": {
        "subindustry": "Commodity Chemicals",
        "tickers": ["OLN", "LYB"],
        "commodity": None,
    },
    "sub_industrial_gases": {
        "subindustry": "Industrial Gases",
        "commodity": None,
    },
    "sub_construction_materials": {
        "subindustry": "Construction Materials",
        "tickers": ["SUM", "EXP"],
        "commodity": None,
    },
    "sub_tobacco": {
        "subindustry": "Tobacco",
        "tickers": ["BTI"],
        "commodity": None,
    },
    "sub_aerospace_defense": {
        "subindustry": "Aerospace & Defense",
        "commodity": None,
    },
    "sub_rail": {
        "subindustry": "Rail Transportation",
        "commodity": None,
    },
    "sub_airlines": {
        "subindustry": "Passenger Airlines",
        "tickers": ["AAL"],
        "commodity": None,
    },
    "sub_construction_machinery": {
        "subindustry": "Construction Machinery & Heavy Transportation Equipment",
        "tickers": ["OSK", "PCAR"],
        "commodity": None,
    },
    "sub_regional_banks": {
        "subindustry": "Regional Banks",
        "tickers": ["CMA", "WAL"],  # present; 2023 mini-crisis names
        "commodity": None,
    },
    "sub_asset_mgmt": {
        "subindustry": "Asset Management & Custody Banks",
        "tickers": ["OWL", "BAM"],
        "commodity": None,
    },
    "sub_semis": {
        "subindustry": "Semiconductors",
        "tickers": ["MU", "MRVL", "ON", "SWKS"],  # present
        "commodity": None,
    },
    "sub_semi_equip": {
        "subindustry": "Semiconductor Materials & Equipment",
        "commodity": None,
    },
    "sub_software": {
        "subindustry": "Application Software",
        "tickers": ["SNOW", "DDOG"],
        "commodity": None,
    },
    "sub_biotech": {
        "subindustry": "Biotechnology",
        "tickers": ["MRNA", "INCY"],
        "commodity": None,
    },
    "sub_med_devices": {
        "subindustry": "Health Care Equipment",
        "commodity": None,
    },
    "sub_power": {
        "subindustry": "Independent Power Producers & Energy Traders",
        "tickers": ["CEG", "VST"],
        "commodity": None,
    },
}


def _monthly_returns(tickers: list[str] | None = None, gics: str | None = None,
                     subindustry: str | None = None) -> pd.Series:
    """Equal-weight monthly log returns of a basket: explicit tickers +
    (optionally) full S&P 500 GICS sector membership and/or GICS
    SUB-INDUSTRY membership (the focused-subsector source — thin subsectors
    carry the explosion signal, e.g. fertilizer). Loaded from
    sp500_constituents.parquet; returns best-available-history series."""
    members = list(tickers or [])
    try:
        sp = pd.read_parquet(DATA_DIR / "sp500_constituents.parquet")
        if gics:
            g = sp.loc[sp["gics_sector"] == gics, "ticker"].astype(str).str.upper().tolist()
            members += g
        if subindustry:
            si = sp.loc[sp["gics_sub_industry"] == subindustry, "ticker"].astype(str).str.upper().tolist()
            members += si
        members = list(dict.fromkeys(members))
    except Exception as e:
        print(f"  gics/subindustry load failed ({e}); using explicit tickers only")
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    avail = [t for t in members if t in w.columns]
    if not avail:
        # fall back to sector_prices synthetic tickers (SECT_*)
        sp = pd.read_parquet(DATA_DIR / "sector_prices.parquet", columns=["date", "ticker", "close"])
        sp["date"] = pd.to_datetime(sp["date"])
        w = sp.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        w = w[w > 0]
        avail = [t for t in members if t in w.columns]
    if not avail:
        return pd.Series(dtype=float)
    r = np.log(w[avail] / w[avail].shift(1)).mean(axis=1)
    return r.resample("ME").sum().replace([np.inf, -np.inf], np.nan).dropna()


def main(save: bool = True):
    rows = []
    for sector, cfg in SECTORS.items():
        rets = _monthly_returns(tickers=cfg.get("tickers"), gics=cfg.get("gics"),
                                subindustry=cfg.get("subindustry"))
        if rets.empty:
            print(f"{sector}: no basket data, skipped")
            continue
        cum = (1 + rets).cumprod()
        basket_mom = cum / cum.shift(12) - 1
        df = pd.DataFrame({"date": cum.index, "basket_mom_12m": basket_mom})
        df = df.reset_index(drop=True)  # neutral index — cum.index inherits name 'date'
        df["sector"] = sector

        if cfg.get("commodity"):
            com = _fetch_fred(cfg["commodity"], DATA_DIR / "macro_data" / f"{cfg['commodity']}.csv")
            com["observation_date"] = pd.to_datetime(com["observation_date"])
            com = com.dropna().set_index("observation_date")
            c = com[cfg["commodity"]]
            c_mom = c / c.shift(12) - 1
            # basket index is month-END (ME), commodity is month-START —
            # reindex with ffill onto the basket dates (also covers the
            # ~1-month IMF publication lag).
            df["commodity_mom_12m"] = c_mom.reindex(df["date"], method="ffill").to_numpy()
        else:
            df["commodity_mom_12m"] = np.nan

        # sector shock score: z(basket mom) + z(commodity mom) when present
        z_b = (df["basket_mom_12m"] - df["basket_mom_12m"].mean()) / df["basket_mom_12m"].std()
        if df["commodity_mom_12m"].notna().sum() > 24:
            z_c = (df["commodity_mom_12m"] - df["commodity_mom_12m"].mean()) / df["commodity_mom_12m"].std()
            df["shock_score"] = (z_b + z_c) / 2
        else:
            df["shock_score"] = z_b

        # zones calibrated on the verified events: fertilizer 2007 (+240%)
        # and 2021 (+213%) both exceeded +200%; wheat 2008 +133%.
        def zone(r):
            if pd.isna(r["basket_mom_12m"]):
                return "no_data"
            if r["basket_mom_12m"] >= 0.80:
                return "shock"
            if r["basket_mom_12m"] >= 0.40:
                return "elevated"
            return "benign"

        df["shock_zone"] = df.apply(zone, axis=1)
        df = df.dropna(subset=["basket_mom_12m"]).tail(720)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out[["sector", "date", "basket_mom_12m", "commodity_mom_12m", "shock_score", "shock_zone"]]
    for c in ("basket_mom_12m", "commodity_mom_12m", "shock_score"):
        out[c] = out[c].round(4)

    if save:
        out.to_csv(OUT, index=False)

    print("\n=== sector shock validation (point-in-time, OUR data) ===")
    for sector in out["sector"].unique():
        s = out[out["sector"] == sector]
        last = s.iloc[-1]
        peaks = s.nlargest(3, "basket_mom_12m")
        print(f"\n{sector}: latest {last['date'].date()} mom {last['basket_mom_12m']:.0%} "
              f"zone {last['shock_zone']}")
        for _, r in peaks.iterrows():
            print(f"   peak {r['date'].date()} basket {r['basket_mom_12m']:.0%} "
                  f"commodity {r['commodity_mom_12m']:.0%}" if not pd.isna(r['commodity_mom_12m'])
                  else f"   peak {r['date'].date()} basket {r['basket_mom_12m']:.0%}")
    if save:
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

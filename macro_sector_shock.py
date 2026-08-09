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
sector_prices) + (optional FRED commodity). A sector shock score =
z(equity 12m momentum) + z(commodity 12m momentum), exactly the
macro_shock recipe (inflation/real-rate legs are macro-wide, not sector-
specific, so they stay in macro_shock.py). Shock zones calibrated on the
verified events above: basket 12m momentum >= +80% = shock, >= +40% =
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

# sector -> (equity basket, optional FRED commodity). Baskets are OUR
# annotated names; commodity series are IMF global prices when they exist.
SECTORS = {
    "farming_inputs": {
        "tickers": ["CF", "MOS", "NTR", "UAN", "IPI", "LXU", "CTVA"],
        "commodity": None,  # no global fertilizer price on FRED; basket carries it
    },
    "farming_outputs": {
        "tickers": ["ADM", "BG", "SYY"],  # grain merchants / food processors
        "commodity": "PWHEAMTUSDM",       # global wheat
    },
    "materials": {
        "tickers": ["SECT_MATERIALS"],    # sector_prices synthetic
        "commodity": "PALLFNFINDEXM",     # IMF all-commodities index
    },
}


def _monthly_returns(tickers: list[str]) -> pd.Series:
    """Equal-weight monthly log returns of a ticker basket (daily_prices
    or sector_prices), best available history per ticker."""
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    avail = [t for t in tickers if t in w.columns]
    if not avail:
        # try sector_prices (SECT_* synthetic tickers)
        sp = pd.read_parquet(DATA_DIR / "sector_prices.parquet", columns=["date", "ticker", "close"])
        sp["date"] = pd.to_datetime(sp["date"])
        w = sp.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        w = w[w > 0]
        avail = [t for t in tickers if t in w.columns]
    if not avail:
        return pd.Series(dtype=float)
    r = np.log(w[avail] / w[avail].shift(1)).mean(axis=1)
    return r.resample("ME").sum().replace([np.inf, -np.inf], np.nan).dropna()


def main(save: bool = True):
    rows = []
    for sector, cfg in SECTORS.items():
        rets = _monthly_returns(cfg["tickers"])
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
            c_mom.index.name = "com_date"
            df = df.merge(c_mom.rename("commodity_mom_12m"), left_on="date", right_index=True, how="left")
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

#!/usr/bin/env python3
"""macro_sector_shock.py — DYNAMIC sector/subsector/factor-group shock signals.

Baskets are NOT a fixed research list. They are built at run time from:

  1. All GICS sectors in sp500_constituents.parquet (current members)
  2. All GICS sub-industries with >= MIN_SUB_N members in daily_prices
  3. All factor_groups.csv groups via factor_group_members.csv (PIT: valid_to null
     or open-ended = current)

Optional IMF commodity series attach by NAME pattern (table-driven mapping),
not by hard-coded ticker lists. Amplifier tickers already in daily_prices are
picked up automatically when they appear in GICS / factor_group membership.

Shock score = z(basket 12m mom) [+ z(commodity 12m mom) when mapped].
Zones: basket 12m mom >= 0.80 shock, >= 0.40 elevated.

Outputs:
  macro_sector_shock.csv — monthly long: basket, basket_kind, date,
      basket_mom_12m, commodity_mom_12m, shock_score, shock_zone, n_members
  basket_members.csv — point-in-time membership: basket, basket_kind, ticker

Usage: python macro_sector_shock.py [--save]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from macro_fragility import _fetch_fred

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "macro_sector_shock.csv"
OUT_MEMBERS = DATA_DIR / "basket_members.csv"
MIN_SUB_N = 2  # min names with price history to keep a sub-industry basket
MIN_OBS_PER_TICKER = 250

# Optional commodity attachment by basket name pattern (NOT ticker lists).
# First match wins. Keys are regexes against basket id (lowercased).
COMMODITY_MAP = [
    (r"copper|sub_copper|industry_copper", "PCOPPUSDM"),
    (r"nickel", "PNICKUSDM"),
    (r"zinc|industrial.?metal", "PZINCUSDM"),
    (r"wheat|grain|farming_output|agricultural product", "PWHEAMTUSDM"),
    (r"corn|maize", "PMAIZMTUSDM"),
    (r"soy", "PSOYBUSDM"),
    (r"sugar", "PSUGAISAUSDM"),
    (r"cotton", "PCOTTINDUSDM"),
    (r"cocoa", "PCOCOUSDM"),
    (r"coffee", "PCOFFOTMUSDM"),
    (r"rubber", "PRUBBUSDM"),
    (r"coal|thermal", "PCOALAUUSDM"),
    (r"uranium", "PURANUSDM"),
    (r"gas|natural.?gas|midstream|oil.?gas storage", "PNGASUSUSDM"),
    (r"^gics_energy$|sector_energy|energy_equit", "PNGASUSUSDM"),
    (r"^gics_materials$|sector_materials|^materials$", "PALLFNFINDEXM"),
]


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _commodity_for(basket_id: str) -> str | None:
    b = basket_id.lower()
    for pat, series in COMMODITY_MAP:
        if re.search(pat, b):
            return series
    return None


def _price_universe() -> set[str]:
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["ticker"])
    return set(p["ticker"].astype(str).str.upper().unique())


def _load_sp500() -> pd.DataFrame:
    sp = pd.read_parquet(DATA_DIR / "sp500_constituents.parquet")
    sp["ticker"] = sp["ticker"].astype(str).str.upper()
    if "current" in sp.columns:
        # keep current if flag exists; else all rows
        cur = sp["current"]
        if cur.dtype == bool:
            sp = sp[cur]
        else:
            sp = sp[cur.astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])]
    return sp


def _build_baskets(have: set[str]) -> dict[str, dict]:
    """Return {basket_id: {kind, tickers, commodity}} — fully dynamic."""
    baskets: dict[str, dict] = {}
    sp = _load_sp500()

    # --- GICS sectors ---
    for sector, g in sp.groupby("gics_sector"):
        tickers = sorted({t for t in g["ticker"] if t in have})
        if len(tickers) < MIN_SUB_N:
            continue
        bid = f"gics_{_slug(sector)}"
        baskets[bid] = {
            "kind": "gics_sector",
            "label": str(sector),
            "tickers": tickers,
            "commodity": _commodity_for(bid + " " + str(sector)),
        }

    # --- GICS sub-industries ---
    for sub, g in sp.groupby("gics_sub_industry"):
        tickers = sorted({t for t in g["ticker"] if t in have})
        if len(tickers) < MIN_SUB_N:
            continue
        bid = f"sub_{_slug(sub)}"
        baskets[bid] = {
            "kind": "gics_subindustry",
            "label": str(sub),
            "tickers": tickers,
            "commodity": _commodity_for(bid + " " + str(sub)),
        }

    # --- factor groups (table-driven, PIT) ---
    fg_path = DATA_DIR / "factor_groups.csv"
    fgm_path = DATA_DIR / "factor_group_members.csv"
    if fg_path.exists() and fgm_path.exists():
        groups = pd.read_csv(fg_path)
        mem = pd.read_csv(fgm_path)
        mem["ticker"] = mem["ticker"].astype(str).str.upper()
        # open-ended membership: valid_to null/NaN/empty
        if "valid_to" in mem.columns:
            vt = mem["valid_to"]
            open_m = vt.isna() | (vt.astype(str).str.strip() == "") | (vt.astype(str).str.lower() == "nan")
            mem = mem[open_m]
        gtype = {}
        if "group" in groups.columns and "group_type" in groups.columns:
            gtype = dict(zip(groups["group"].astype(str), groups["group_type"].astype(str)))
        for grp, g in mem.groupby("group"):
            tickers = sorted({t for t in g["ticker"] if t in have})
            if len(tickers) < MIN_SUB_N:
                continue
            bid = f"fg_{_slug(grp)}"
            baskets[bid] = {
                "kind": f"factor_group:{gtype.get(str(grp), 'group')}",
                "label": str(grp),
                "tickers": tickers,
                "commodity": _commodity_for(bid + " " + str(grp)),
            }

    return baskets


def _monthly_returns(tickers: list[str]) -> pd.Series:
    """Equal-weight monthly log returns for a ticker list."""
    p = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    p["date"] = pd.to_datetime(p["date"])
    # restrict early
    p = p[p["ticker"].isin(tickers)]
    if p.empty:
        return pd.Series(dtype=float)
    w = p.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    # drop tickers with thin history
    keep = [c for c in w.columns if w[c].notna().sum() >= MIN_OBS_PER_TICKER]
    if len(keep) < MIN_SUB_N:
        keep = list(w.columns)
    if not keep:
        return pd.Series(dtype=float)
    r = np.log(w[keep] / w[keep].shift(1)).mean(axis=1)
    return r.resample("ME").sum().replace([np.inf, -np.inf], np.nan).dropna()


def main(save: bool = True):
    print("macro_sector_shock: building DYNAMIC baskets from GICS + factor_groups…")
    have = _price_universe()
    print(f"  price universe: {len(have)} tickers")
    baskets = _build_baskets(have)
    print(f"  baskets: {len(baskets)} "
          f"(gics_sector={sum(1 for b in baskets.values() if b['kind']=='gics_sector')}, "
          f"sub={sum(1 for b in baskets.values() if b['kind']=='gics_subindustry')}, "
          f"fg={sum(1 for b in baskets.values() if str(b['kind']).startswith('factor_group'))})")

    # membership dump
    mem_rows = []
    for bid, cfg in baskets.items():
        for t in cfg["tickers"]:
            mem_rows.append({
                "basket": bid,
                "basket_kind": cfg["kind"],
                "label": cfg["label"],
                "ticker": t,
                "commodity": cfg.get("commodity") or "",
            })
    mem_df = pd.DataFrame(mem_rows)

    # commodity cache
    com_cache: dict[str, pd.Series] = {}

    rows = []
    for bid, cfg in sorted(baskets.items()):
        rets = _monthly_returns(cfg["tickers"])
        if rets.empty or len(rets) < 14:
            continue
        cum = (1 + rets).cumprod()
        basket_mom = cum / cum.shift(12) - 1
        df = pd.DataFrame({"date": cum.index, "basket_mom_12m": basket_mom})
        df = df.reset_index(drop=True)
        df["basket"] = bid
        df["basket_kind"] = cfg["kind"]
        df["label"] = cfg["label"]
        df["n_members"] = len(cfg["tickers"])

        series = cfg.get("commodity")
        if series:
            if series not in com_cache:
                try:
                    com = _fetch_fred(series, DATA_DIR / "macro_data" / f"{series}.csv")
                    com["observation_date"] = pd.to_datetime(com["observation_date"])
                    com = com.dropna().set_index("observation_date")
                    c = com[series]
                    com_cache[series] = (c / c.shift(12) - 1)
                except Exception as e:
                    print(f"  commodity {series} fail ({e})")
                    com_cache[series] = pd.Series(dtype=float)
            c_mom = com_cache[series]
            if not c_mom.empty:
                df["commodity_mom_12m"] = c_mom.reindex(df["date"], method="ffill").to_numpy()
            else:
                df["commodity_mom_12m"] = np.nan
        else:
            df["commodity_mom_12m"] = np.nan

        z_b = (df["basket_mom_12m"] - df["basket_mom_12m"].mean()) / (df["basket_mom_12m"].std() or 1.0)
        if df["commodity_mom_12m"].notna().sum() > 24:
            z_c = (df["commodity_mom_12m"] - df["commodity_mom_12m"].mean()) / (df["commodity_mom_12m"].std() or 1.0)
            df["shock_score"] = (z_b + z_c) / 2
        else:
            df["shock_score"] = z_b

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

    if not rows:
        print("no baskets produced")
        return

    out = pd.concat(rows, ignore_index=True)
    out = out[[
        "basket", "basket_kind", "label", "date", "n_members",
        "basket_mom_12m", "commodity_mom_12m", "shock_score", "shock_zone",
    ]]
    for c in ("basket_mom_12m", "commodity_mom_12m", "shock_score"):
        out[c] = out[c].round(4)

    if save:
        out.to_csv(OUT, index=False)
        mem_df.to_csv(OUT_MEMBERS, index=False)

    # report: latest zone distribution + top shocks
    latest = out.sort_values("date").groupby("basket", as_index=False).tail(1)
    print("\n=== dynamic basket shock (latest) ===")
    print(latest["shock_zone"].value_counts().to_string())
    shocks = latest[latest["shock_zone"].isin(["shock", "elevated"])].sort_values(
        "basket_mom_12m", ascending=False
    )
    print(f"\nshock/elevated ({len(shocks)}):")
    for _, r in shocks.head(20).iterrows():
        print(f"  {r['basket']:40s} {r['shock_zone']:8s} mom {r['basket_mom_12m']:6.0%}  n={int(r['n_members'])}")
    if save:
        print(f"\nWrote {OUT} ({len(out)} rows)")
        print(f"Wrote {OUT_MEMBERS} ({len(mem_df)} membership rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    main(save=True)

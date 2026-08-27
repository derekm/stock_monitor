#!/usr/bin/env python3
"""
stress_dual_pass.py — Stress-test dual-pass inclusion criteria.

Varies ROE/ROIC/D/E/EV/P/B/MCA thresholds and reports how many names pass.
Also runs one-leg relaxation sensitivity.

Usage:
  python stress_dual_pass.py
  python stress_dual_pass.py --save
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "dual_pass_stress.parquet"
OUT_SENS = DATA_DIR / "dual_pass_sensitivity.parquet"

ERP_BASE = 0.0423  # Damodaran implied ERP Jan 2026
TAX_RATE = 0.21
SYNTHETIC_RATING = [
    (8.5, "Aaa", 0.0040), (6.5, "Aa", 0.0070), (5.5, "A", 0.0090),
    (4.25, "Baa", 0.0150), (3.0, "Ba", 0.0250), (2.0, "B", 0.0400),
    (1.5, "Caa", 0.0600), (0.0, "Ca", 0.1000),
]
SECTOR_BETAS = {
    "Technology": 1.15, "Communication Services": 1.10,
    "Consumer Discretionary": 1.20, "Consumer Staples": 0.80,
    "Health Care": 0.85, "Financials": 1.10, "Industrials": 1.05,
    "Materials": 1.10, "Energy": 1.15, "Utilities": 0.70,
    "Real Estate": 0.90, "ETF": 1.00,
}


def latest_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def synthetic_rating_from_coverage(interest_coverage):
    """Return (rating, default_spread) from interest coverage per Damodaran table."""
    if pd.isna(interest_coverage) or interest_coverage <= 0:
        return "Ca", 0.1000
    for threshold, rating, spread in SYNTHETIC_RATING:
        if interest_coverage >= threshold:
            return rating, spread
    return "Ca", 0.1000


def compute_wacc_scenarios(fund: pd.DataFrame, rf: float = 0.0418) -> dict[str, pd.DataFrame]:
    """Compute WACC per ticker under ERP scenarios.

    Macro volatility shifts ERP:
      - low_vol: ERP compressed (risk appetite high) -> lower WACC -> higher fair value
      - base: current Damodaran implied ERP
      - high_vol: ERP expanded -> higher WACC -> lower fair value
      - crisis: ERP spikes -> much higher WACC

    Returns dict of scenario_name -> DataFrame with WACC per ticker.
    """
    scenarios = {
        "low_vol": rf + (ERP_BASE - 0.02),  # ERP = 2.23%
        "base": rf + ERP_BASE,               # ERP = 4.23%
        "high_vol": rf + (ERP_BASE + 0.025), # ERP = 6.73%
        "crisis": rf + (ERP_BASE + 0.04),    # ERP = 8.23%
    }

    results = {}
    for scenario_name, erp_val in scenarios.items():
        rows = []
        for _, row in fund.iterrows():
            ticker = row.get("ticker")
            sector = row.get("sector", "Technology")
            sector_beta = SECTOR_BETAS.get(sector, 1.0)

            cost_of_equity = rf + sector_beta * erp_val

            ic = row.get("interest_coverage")
            rating, default_spread = synthetic_rating_from_coverage(ic)
            cost_of_debt = rf + default_spread
            after_tax_cost_of_debt = cost_of_debt * (1 - TAX_RATE)

            de = row.get("debt_to_equity")
            market_cap = row.get("market_cap")

            if pd.notna(de) and de > 0 and pd.notna(market_cap) and market_cap > 0:
                E = market_cap
                D = de * market_cap
                if D > 0 and E > 0:
                    w_e = E / (D + E)
                    w_d = D / (D + E)
                else:
                    w_e, w_d = 1.0, 0.0
            else:
                w_e, w_d = 1.0, 0.0

            wacc = cost_of_equity * w_e + after_tax_cost_of_debt * w_d

            rows.append({
                "ticker": ticker,
                "sector": sector,
                "erp_scenario": erp_val,
                "cost_of_equity": round(cost_of_equity, 6),
                "wacc": round(wacc, 6),
                "weight_equity": round(w_e, 4),
                "weight_debt": round(w_d, 4),
            })
        results[scenario_name] = pd.DataFrame(rows)
    return results


def count_cheap_by_wacc(fund: pd.DataFrame, wacc_df: pd.DataFrame, min_excess: float = 0.02) -> tuple[int, list]:
    """Count tickers where ROE > WACC + min_excess (value creators under Damodaran)."""
    merged = fund.merge(wacc_df[["ticker", "wacc", "cost_of_equity"]], on="ticker", how="left")
    merged["excess_return"] = merged["roe"] - merged["cost_of_equity"]
    mask = (merged["excess_return"] >= min_excess) & merged["wacc"].notna()
    return int(mask.sum()), merged.loc[mask, "ticker"].tolist()


def risk_of_names(df, tickers, prices_path=DATA_DIR / "daily_prices/"):
    """Portfolio EW risk metrics for a ticker list."""
    if not tickers:
        return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"))
    try:
        prices = pd.read_parquet(prices_path, columns=["date", "ticker", "close"])
        prices["date"] = pd.to_datetime(prices["date"])
        wide = prices[prices.ticker.isin(tickers)].pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
        rets = np.log(wide / wide.shift(1)).dropna(how="all")
        if rets.empty:
            return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"))
        ew = rets.mean(axis=1)
        port_vol = float(ew.std() * np.sqrt(252))
        cum = ew.cumsum()
        max_dd = float((np.exp(cum) / np.exp(cum).cummax() - 1).min())
        name_vol = rets.std() * np.sqrt(252)
        mkt = rets.mean(axis=1)
        betas = []
        for c in rets.columns:
            cov = np.cov(rets[c].dropna().align(mkt, join="inner")[0], rets[c].dropna().align(mkt, join="inner")[1])
            # simpler:
            aligned = pd.concat([rets[c], mkt], axis=1, keys=["a","m"]).dropna()
            if len(aligned) > 20 and aligned["m"].var() > 0:
                betas.append(float(aligned.cov().iloc[0,1] / aligned["m"].var()))
        return dict(
            port_vol=port_vol,
            port_max_dd=max_dd,
            avg_beta=float(np.mean(betas)) if betas else float("nan"),
            avg_name_vol=float(name_vol.mean()) if len(name_vol) else float("nan"),
            n_names=len(tickers),
        )
    except Exception as e:
        return dict(port_vol=float("nan"), port_max_dd=float("nan"), avg_beta=float("nan"), avg_name_vol=float("nan"), error=str(e))

def count_pass(df, roe_min, roic_min, de_max, ev_max, pb_max, mca_max):
    m = (
        (df["roe"] >= roe_min) & (df["roic"] >= roic_min) & (df["debt_to_equity"] <= de_max)
        & (df["ev_ebitda"] <= ev_max) & (df["pb_ratio"] <= pb_max) & (df["mktcap_to_assets"] <= mca_max)
    )
    return int(m.sum()), df.loc[m, "ticker"].tolist()


def run(save: bool = True):
    df = latest_fund()
    base = dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9.0, pb_max=1.5, mca_max=0.5)
    n0, t0 = count_pass(df, **base)
    print(f"Base dual-pass: {n0} names → {t0}")

    # Grid stress
    grid = {
        "roe_min": [0.10, 0.12, 0.15, 0.18, 0.20],
        "roic_min": [0.10, 0.12, 0.15, 0.18],
        "de_max": [0.5, 1.0, 1.5, 2.0],
        "ev_max": [7.0, 9.0, 12.0, 15.0],
        "pb_max": [1.0, 1.5, 2.0, 3.0],
        "mca_max": [0.3, 0.5, 0.8, 1.2],
    }
    rows = []
    # one-parameter-at-a-time from base
    for param, values in grid.items():
        for val in values:
            kw = dict(base)
            kw[param] = val
            n, tickers = count_pass(df, **kw)
            risk = risk_of_names(df, tickers)
            rows.append({
                "mode": "one_at_a_time",
                "param": param,
                "value": str(val),
                "n_pass": n,
                "delta_vs_base": n - n0,
                "tickers": ",".join(tickers[:20]),
                **{k: risk.get(k) for k in ("port_vol","port_max_dd","avg_beta","avg_name_vol")},
            })
    # joint relaxed / tight scenarios
    scenarios = [
        ("tight", dict(roe_min=0.18, roic_min=0.18, de_max=0.5, ev_max=7, pb_max=1.0, mca_max=0.3)),
        ("base", base),
        ("relaxed_quality", dict(roe_min=0.12, roic_min=0.12, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("relaxed_value", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=12, pb_max=2.0, mca_max=0.8)),
        ("relaxed_both", dict(roe_min=0.12, roic_min=0.12, de_max=1.5, ev_max=12, pb_max=2.0, mca_max=0.8)),
        ("buffett_fair", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=15, pb_max=3.0, mca_max=1.5)),
    ]
    for name, kw in scenarios:
        n, tickers = count_pass(df, **kw)
        risk = risk_of_names(df, tickers)
        rows.append({
            "mode": "scenario",
            "param": name,
            "value": str(kw),
            "n_pass": n,
            "delta_vs_base": n - n0,
            "tickers": ",".join(tickers[:25]),
            **{k: risk.get(k) for k in ("port_vol","port_max_dd","avg_beta","avg_name_vol")},
        })
        print(f"  scenario {name:16s}  n={n:3d}  ({n-n0:+d})  vol={risk.get('port_vol') and risk['port_vol']*100:.1f}%  {tickers[:6]}")

    # leave-one-leg-out sensitivity
    sens = []
    legs = [
        ("drop_roe", dict(roe_min=-9, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_roic", dict(roe_min=0.15, roic_min=-9, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_de", dict(roe_min=0.15, roic_min=0.15, de_max=99, ev_max=9, pb_max=1.5, mca_max=0.5)),
        ("drop_ev", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=99, pb_max=1.5, mca_max=0.5)),
        ("drop_pb", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=99, mca_max=0.5)),
        ("drop_mca", dict(roe_min=0.15, roic_min=0.15, de_max=1.0, ev_max=9, pb_max=1.5, mca_max=99)),
    ]
    print("\n=== Leave-one-leg-out ===")
    for name, kw in legs:
        n, tickers = count_pass(df, **kw)
        new = sorted(set(tickers) - set(t0))
        sens.append({"dropped_leg": name, "n_pass": n, "delta": n - n0, "new_tickers": ",".join(new[:15])})
        print(f"  {name:12s} n={n:3d} (+{n-n0}) new={new[:10]}")

    # ── Damodaran ERP scenario analysis ────────────────────────────────────
    print("\n=== Damodaran ERP scenario analysis (macro vol → ERP shift) ===")
    try:
        # Load sector metadata
        stocks_path = DATA_DIR / "monitored_stocks.parquet"
        stocks = pd.read_parquet(stocks_path) if stocks_path.exists() else pd.DataFrame()
        if not stocks.empty and "sector" in stocks.columns:
            sec_map = stocks.set_index("ticker")["sector"].to_dict()
            df["sector"] = df["ticker"].map(sec_map).fillna("Technology")
        else:
            df["sector"] = "Technology"

        wacc_scenarios = compute_wacc_scenarios(df)
        scenario_rows = []
        for scenario_name, wacc_df in wacc_scenarios.items():
            n_val, cheap_tickers = count_cheap_by_wacc(df, wacc_df)
            risk = risk_of_names(df, cheap_tickers)
            scenario_rows.append({
                "mode": "erp_scenario",
                "param": scenario_name,
                "value": f"ERP={wacc_scenarios[scenario_name]['erp_scenario'].iloc[0]:.2%}",
                "n_pass": n_val,
                "delta_vs_base": n_val - n0,
                "tickers": ",".join(cheap_tickers[:25]),
                **{k: risk.get(k) for k in ("port_vol", "port_max_dd", "avg_beta", "avg_name_vol")},
            })
            print(f"  ERP scenario {scenario_name:10s}: {n_val:3d} value creators (ROE > COE+2%) ({n_val-n0:+d}) vol={risk.get('port_vol') and risk['port_vol']*100:.1f}%  {cheap_tickers[:6]}")

        # Joint: dual-pass + Damodaran value creation
        wacc_base = wacc_scenarios["base"]
        for scenario_name in ["low_vol", "base", "high_vol", "crisis"]:
            wacc_scen = wacc_scenarios[scenario_name]
            n_creators, creators = count_cheap_by_wacc(df, wacc_scen)
            dual_pass = set(count_pass(df, **base)[1])
            joint = sorted(set(creators) & dual_pass)
            print(f"  Joint dual-pass + Damodaran {scenario_name:10s}: {len(joint)} names (dual-pass AND value creator)")
        rows.extend(scenario_rows)
    except Exception as e:
        print(f"  ERP scenario analysis skipped: {e}")

    out = pd.DataFrame(rows)
    sens_df = pd.DataFrame(sens)
    if save:
        out.to_parquet(OUT)
        sens_df.to_parquet(OUT_SENS)
        print(f"Wrote {OUT}\nWrote {OUT_SENS}")
    return out, sens_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()

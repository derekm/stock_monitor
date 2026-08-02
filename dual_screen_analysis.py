#!/usr/bin/env python3
"""
dual_screen_analysis.py — Why Buffett+Trifecta dual pass is rare; external candidates.

Explains the structural tension between:
  Quality: high ROE/ROIC, low debt  → often high P/B and EV/EBITDA
  Value trifecta: cheap on EV/EBITDA, P/B, MktCap/Assets → often cyclical/low-ROE

Lists external (not currently monitored) tickers that *could* approach dual pass
with approximate public-market characteristics (illustrative — verify live data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT = DATA_DIR / "dual_screen_gap.csv"
OUT_EXT = DATA_DIR / "dual_screen_external_candidates.csv"

# Illustrative external names sometimes cited near "quality at a reasonable price"
# Approximate — must be verified before use
EXTERNAL_CANDIDATES = [
    # ticker, name, roe, roic, de, pb, ev_ebitda, mca, note
    ("MU", "Micron", 0.16, 0.12, 0.3, 1.8, 6.5, 0.9, "Memory cyclical — ROE swings; sometimes cheap on EV"),
    ("HPQ", "HP Inc", 0.20, 0.15, 0.8, 1.4, 7.0, 0.6, "PC hardware — occasionally QARP-like"),
    ("TPR", "Tapestry", 0.22, 0.14, 0.7, 2.0, 8.0, 0.8, "Consumer brand — check leverage"),
    ("AFL", "Aflac", 0.16, 0.12, 0.4, 1.5, 8.5, 0.15, "Insurance — book value relevant"),
    ("MET", "MetLife", 0.12, 0.09, 0.5, 1.0, 7.0, 0.08, "Near dual on book; ROE borderline"),
    ("PRU", "Prudential", 0.11, 0.08, 0.5, 0.9, 6.5, 0.07, "Similar to MET"),
    ("VLO", "Valero", 0.20, 0.12, 0.4, 1.5, 5.0, 0.7, "Refiner cyclical — high ROE in upcycle + low EV"),
    ("PSX", "Phillips 66", 0.18, 0.11, 0.5, 1.6, 6.0, 0.75, "Refiner QARP in parts of cycle"),
    ("MPC", "Marathon Petroleum", 0.22, 0.14, 0.5, 1.7, 5.5, 0.8, "Refiner"),
    ("CNC", "Centene", 0.12, 0.08, 0.7, 1.2, 7.5, 0.5, "Managed care value"),
    ("CI", "Cigna", 0.14, 0.10, 0.6, 1.8, 8.0, 0.7, "Near quality/value border"),
    ("KR", "Kroger", 0.22, 0.10, 1.5, 2.5, 7.0, 0.4, "ROE ok; leverage elevated"),
    ("SYF", "Synchrony", 0.18, 0.12, 1.2, 1.1, 5.0, 0.15, "Consumer finance — ROE + low P/B"),
    ("CFG", "Citizens Financial", 0.10, 0.08, 0.5, 0.8, 7.0, 0.1, "Regional bank book-value value"),
    ("KEY", "KeyCorp", 0.09, 0.07, 0.5, 0.9, 8.0, 0.1, "Regional bank"),
    ("NUE", "Nucor", 0.18, 0.14, 0.3, 1.6, 6.0, 1.0, "Steel — ROIC solid in mid-cycle"),
    ("STLD", "Steel Dynamics", 0.20, 0.15, 0.3, 1.8, 5.5, 1.1, "Steel"),
    ("DVN", "Devon Energy", 0.18, 0.12, 0.5, 1.5, 4.5, 0.9, "E&P mid-cycle"),
    ("FANG", "Diamondback", 0.16, 0.12, 0.4, 1.4, 5.0, 0.9, "E&P"),
    ("CEIX", "Consol Energy", 0.25, 0.18, 0.3, 1.5, 4.0, 1.0, "Coal/thermal — high ROE when prices firm"),
]


def analyze():
    fund = pd.read_parquet(FUND)
    if "as_of_date" in fund.columns:
        fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
    stocks = pd.read_parquet(STOCKS)
    monitored = set(stocks["ticker"])

    q = (fund["roe"] >= 0.15) & (fund["roic"] >= 0.15) & (fund["debt_to_equity"] <= 1.0)
    v = (fund["ev_ebitda"] <= 9) & (fund["pb_ratio"] <= 1.5) & (fund["mktcap_to_assets"] <= 0.5)

    print("=== Dual-screen gap (monitored universe) ===")
    print(f"Buffett quality pass: {q.sum()}")
    print(f"Value trifecta pass:  {v.sum()}")
    print(f"BOTH:                 {(q & v).sum()}")

    b = fund[q]
    t = fund[v]
    print("\nWhy the gap?")
    print("  Quality names — median P/B / EV/EBITDA / MCA:")
    print(f"    P/B={b.pb_ratio.median():.2f}  EV/EBITDA={b.ev_ebitda.median():.1f}  "
          f"MCA={b.mktcap_to_assets.median():.2f}")
    print("  Trifecta names — median ROE / ROIC / D/E:")
    print(f"    ROE={t.roe.median():.1%}  ROIC={t.roic.median():.1%}  D/E={t.debt_to_equity.median():.2f}")
    print("""
  Structural tension:
  • High ROE/ROIC compounders are priced for quality → P/B and EV/EBITDA rise above trifecta caps.
  • Trifecta cheapness often appears in cyclicals, banks, or impaired earners → ROE/ROIC < 15%.
  • Buffett’s actual practice is closer to 'wonderful company at fair price' than pure trifecta;
    he accepts higher multiples when ROE is durable and capital allocation is excellent.
""")

    # gap table
    rows = []
    for _, r in fund.iterrows():
        rows.append({
            "ticker": r["ticker"],
            "buffett_pass": bool(r.ticker in fund[q]["ticker"].values),
            "trifecta_pass": bool(r.ticker in fund[v]["ticker"].values),
            "roe": r.get("roe"), "roic": r.get("roic"), "debt_to_equity": r.get("debt_to_equity"),
            "pb_ratio": r.get("pb_ratio"), "ev_ebitda": r.get("ev_ebitda"),
            "mktcap_to_assets": r.get("mktcap_to_assets"),
            "gap": (
                "dual" if (r.ticker in set(fund[q].ticker) and r.ticker in set(fund[v].ticker)) else
                "quality_only" if r.ticker in set(fund[q].ticker) else
                "value_only" if r.ticker in set(fund[v].ticker) else
                "neither"
            ),
        })
    gap = pd.DataFrame(rows)
    gap.to_csv(OUT, index=False)

    print("=== External candidates (NOT in monitored set) — illustrative ===")
    print("Approximate characteristics; verify with live data before any decision.\n")
    ext = []
    for t, name, roe, roic, de, pb, ev, mca, note in EXTERNAL_CANDIDATES:
        if t in monitored:
            status = "already_monitored"
        else:
            status = "not_monitored"
        buffett = roe >= 0.15 and roic >= 0.15 and de <= 1.0
        trif = ev is not None and ev <= 9 and pb <= 1.5 and mca <= 0.5
        near = (roe >= 0.12 and roic >= 0.10 and de <= 1.2 and
                ev is not None and ev <= 10 and pb <= 2.0 and mca <= 0.9)
        ext.append({
            "ticker": t, "name": name, "status": status,
            "roe": roe, "roic": roic, "debt_to_equity": de,
            "pb_ratio": pb, "ev_ebitda": ev, "mktcap_to_assets": mca,
            "buffett_pass_approx": buffett, "trifecta_pass_approx": trif,
            "near_dual_approx": near or (buffett and trif),
            "note": note,
        })
    edf = pd.DataFrame(ext)
    edf.to_csv(OUT_EXT, index=False)
    dual = edf[(edf.buffett_pass_approx) & (edf.trifecta_pass_approx)]
    near = edf[edf.near_dual_approx & ~((edf.buffett_pass_approx) & (edf.trifecta_pass_approx))]
    print("Approx dual pass:")
    print(dual[["ticker","name","roe","roic","pb_ratio","ev_ebitda","mktcap_to_assets"]].to_string(index=False)
          if len(dual) else "  (none strict)")
    print("\nNear dual (relaxed borders):")
    print(near[["ticker","name","roe","roic","pb_ratio","ev_ebitda","note"]].to_string(index=False))
    print(f"\nWrote {OUT}\nWrote {OUT_EXT}")


if __name__ == "__main__":
    analyze()

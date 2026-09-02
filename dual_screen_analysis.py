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
OUT = DATA_DIR / "dual_screen_gap.parquet"
OUT_EXT = DATA_DIR / "dual_screen_external_candidates.parquet"

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


def gray_vogel_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 2 item 9: Alpha Architect QV/QM long-short vs TMI.

    QV per Gray/Carlisle: EV/EBITDA cheap quintile + GP/A top quintile +
    low leverage (D/E <= median), all PIT on the filing calendar.
    QM: 12-1 momentum + nm_quality (>=2 legs: GP/A top quintile, low
    accruals, safe leverage).

    Monthly-rebalance, equal-weight top/bottom quintile, 10 bps per side,
    same calendar as item 8. Bar: QV∩NM >= 80% on names with GP/A, and net
    spread vs dual-pass is quoted (not overlap theater).
    """
    import numpy as np
    from macro_sector_shock import _load_price_matrix

    fund = pd.read_parquet(FUND)
    fund = fund.sort_values(["as_of_date", "ticker"])
    eveb = fund.pivot(index="as_of_date", columns="ticker", values="ev_ebitda")
    de = fund.pivot(index="as_of_date", columns="ticker", values="debt_to_equity")
    w = _load_price_matrix()
    lp = np.log(w.replace(0, np.nan))
    dates = pd.DatetimeIndex(lp.index)
    month_ends = dates[dates.is_month_end]

    # GP/A wide panel (filing calendar; columns already tickers)
    gp = pd.read_parquet(DATA_DIR / "novymarx_gross_profitability.parquet")
    gp.columns = gp.columns.astype(str).str.upper()
    # accruals + asset growth + D/E legs for nm_quality, same panel source
    ac = pd.read_parquet(DATA_DIR / "novymarx_accruals.parquet")
    ag = pd.read_parquet(DATA_DIR / "novymarx_asset_growth.parquet")
    dev = pd.read_parquet(DATA_DIR / "novymarx_debt_to_equity.parquet")
    for p in (ac, ag, dev):
        p.columns = p.columns.astype(str).str.upper()

    # Pre-ffill once per panel (PIT: last filing value carried forward), then
    # each month-end is a single row lookup — no per-date ffill over 9k cols.
    evebF = eveb.ffill()
    deF = de.ffill()
    gpF = gp.ffill()
    acF = ac.ffill()
    agF = ag.ffill()
    devF = dev.ffill()

    def pit(row_date, wide_full):
        """Cross-section at row_date (last ffill'd row <= row_date)."""
        idx = pd.DatetimeIndex(wide_full.index)
        ww = wide_full.loc[idx <= pd.Timestamp(row_date)]
        if ww.empty:
            return pd.Series(dtype=float)
        return ww.iloc[-1]

    frames = []
    for j, t in enumerate(month_ends):
        i = dates.get_loc(t)
        if i < 252:
            continue
        part = pd.DataFrame({"ticker": lp.columns})
        part["date"] = t.date()
        eveb_t = pit(t, evebF)
        de_t = pit(t, deF)
        gp_t = pit(t, gpF)
        ac_t = pit(t, acF)
        ag_t = pit(t, agF)
        dev_t = pit(t, devF)
        # QV legs (cross-sectional quintiles on today's filing cross-section)
        ev_q = eveb_t.rank(pct=True)                       # low = cheap
        gp_q = gp_t.rank(pct=True)                         # high = profitable
        de_med = de_t.dropna().median()
        qv_cheap = ev_q <= 0.20
        qv_gp = gp_q >= 0.80
        qv_lev = de_t.notna() & (de_t <= de_med)
        part["qv_leg_cheap"] = qv_cheap.reindex(part["ticker"]).values
        part["qv_leg_gp"] = qv_gp.reindex(part["ticker"]).values
        part["qv_leg_lev"] = qv_lev.reindex(part["ticker"]).values
        part["qv_flag"] = (
            part["qv_leg_cheap"] & part["qv_leg_gp"] & part["qv_leg_lev"]
        )
        # QM legs: 12-1 momentum + nm_quality >=2 legs
        mom = (lp.iloc[i - 21] - lp.iloc[i - 252]).rename("mom_12_1")
        ag_w = pd.Series(np.clip(ag_t, -1.0, 1.0), index=ag_t.index)
        nm = pd.DataFrame({
            "gp_q": gp_q, "ag_q": 1 - ag_w.rank(pct=True),
            "ac_q": 1 - ac_t.rank(pct=True), "de_q": 1 - dev_t.rank(pct=True),
        })
        nm["legs"] = nm[["gp_q", "ag_q", "ac_q", "de_q"]].notna().sum(axis=1)
        nm_score = nm[["gp_q", "ag_q", "ac_q", "de_q"]].mean(axis=1)
        nm_qual = (nm_score >= 0.5) & (nm["legs"] >= 2)
        part["qm_mom"] = mom.reindex(part["ticker"]).values
        part["qm_nm"] = nm_qual.reindex(part["ticker"]).values
        part["qm_flag"] = part["qm_nm"] & (part["qm_mom"].rank(pct=True) >= 0.5)
        # forward monthly return
        if j + 1 < len(month_ends):
            i2 = dates.get_loc(month_ends[j + 1])
            part["fwd_ret"] = (lp.iloc[i2] - lp.iloc[i]).values
            frames.append(part)
    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    tmi["date"] = pd.to_datetime(tmi["date"]).dt.date
    tmi = tmi.set_index("date")["ret_net"]
    panel["tmi"] = panel["date"].map(lambda d: tmi.get(d, np.nan))

    # Long-short quintile per date, 10 bps per side
    out = []
    for d, g in panel.groupby("date"):
        row = {"date": d}
        s = g.dropna(subset=["fwd_ret"]).copy()
        if len(s) < 40:
            continue
        for sig, qcol in (("qv_flag", "qv_flag"), ("qm_flag", "qm_flag")):
            sub = s.dropna(subset=[qcol])
            if len(sub) < 20:
                row[sig] = np.nan
                continue
            long_r = float(sub.loc[sub[qcol], "fwd_ret"].mean())
            short_r = float(sub.loc[~sub[qcol], "fwd_ret"].mean())
            row[sig] = (long_r - short_r) - 0.002
        row["tmi"] = float(g["tmi"].mean())
        out.append(row)
    ls = pd.DataFrame(out).set_index("date")
    ann = {}
    for sig in ("qv_flag", "qm_flag"):
        s = ls[sig].dropna()
        ann[sig] = {"net_ann": float(s.mean() * 12) if len(s) else np.nan,
                    "n_months": int(len(s)),
                    "gross_ann": float((s + 0.002).mean() * 12) if len(s) else np.nan}
    ann_df = pd.DataFrame(ann).T
    ann_df.index.name = "signal"
    ann_df = ann_df.reset_index()
    ann_df["vs_tmi_net"] = ann_df["net_ann"] - float(ls["tmi"].mean() * 12)

    # QV∩NM bar on latest date (names with GP/A)
    last = panel[panel["date"] == panel["date"].max()]
    gpl = last[last["qv_leg_gp"] & last["qv_flag"]]
    if len(gpl):
        overlap = float(last[last["qv_flag"] & last["qm_nm"]].shape[0]) / max(1, int(gpl.shape[0]))
    else:
        overlap = np.nan
    ann_df.attrs["qv_nm_overlap"] = overlap
    return ls, ann_df


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
    gap.to_parquet(OUT)

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
    edf.to_parquet(OUT_EXT)
    dual = edf[(edf.buffett_pass_approx) & (edf.trifecta_pass_approx)]
    near = edf[edf.near_dual_approx & ~((edf.buffett_pass_approx) & (edf.trifecta_pass_approx))]
    print("Approx dual pass:")
    print(dual[["ticker","name","roe","roic","pb_ratio","ev_ebitda","mktcap_to_assets"]].to_string(index=False)
          if len(dual) else "  (none strict)")
    print("\nNear dual (relaxed borders):")
    print(near[["ticker","name","roe","roic","pb_ratio","ev_ebitda","note"]].to_string(index=False))
    print(f"\nWrote {OUT}\nWrote {OUT_EXT}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray-vogel", action="store_true",
                    help="Phase 2 item 9: QV/QM long-short vs TMI (PIT, 10 bps)")
    args = ap.parse_args()
    if args.gray_vogel:
        import numpy as np
        ls, ann = gray_vogel_backtest()
        pd.set_option("display.width", 200)
        print("\n=== Gray/Vogel QV/QM (monthly rebalance, EW flag LS, 10 bps/side, vs TMI) ===")
        print(ann.to_string(index=False))
        ov = ann.attrs.get("qv_nm_overlap")
        if ov is not None:
            print(f"\nBAR QV∩NM (on names with GP/A, latest date): {ov:.1%} -> "
                  f"{'PASS' if ov >= 0.80 else 'FAIL'} (bar >= 80%)")
        qv = float(ann.loc[ann["signal"] == "qv_flag", "net_ann"].iloc[0])
        qm = float(ann.loc[ann["signal"] == "qm_flag", "net_ann"].iloc[0])
        print(f"\nQV net {qv:+.1%}/yr | QM net {qm:+.1%}/yr (annualized, vs TMI spread listed above)")
        ann.to_parquet(DATA_DIR / "gray_vogel_ls.parquet", index=False)
        print("Wrote gray_vogel_ls.parquet")
        return
    analyze()


if __name__ == "__main__":
    main()

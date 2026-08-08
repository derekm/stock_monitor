#!/usr/bin/env python3
"""fragility_screen.py — per-name fragility index (the Taleb layer).

Why: cheap is not enough. A name can be cheap AND fragile — fragile names get
destroyed in the tails regardless of valuation. Taleb's fragility = sensitivity
to volatility/vol-of-vol. This screen combines the repo's own data:

1. leverage: debt_to_equity (fundamentals) — high D/E = fragile to rate/vol shocks
2. mktcap_to_assets (fundamentals) — low coverage = fragile
3. interest_coverage (fundamentals) — below 3 = fragile
4. IV skew (options_skew.csv) — steep skew = the market prices downside fear
   (the market's own fragility gauge, per Taleb)
5. illiquidity proxy: avg daily dollar volume (daily_prices) — low = fragile
   (can't exit in a gap)
6. earnings variability (fundamentals ROE std via estimate_revisions or
   earnings history) — unstable earnings = fragile
7. gap share of variance (gap_risk.csv) — risk arriving overnight = fragile
   to gap-through-stops
8. tail alpha (tail_index.csv) — low alpha = fatter tails

Each input is converted to a fragility percentile (0..1, higher = worse),
then combined with fixed weights. Outputs a fragility score, a percentile rank
and a FRAGILE veto flag (score in top decile) for the inclusion gates.

Outputs:
  fragility_screen.csv   per-ticker component percentiles, composite
                         fragility score, percentile, fragile flag
Reads: fundamentals.parquet, options_skew.csv, gap_risk.csv, tail_index.csv,
       daily_prices.parquet (volume).
Usage: python fragility_screen.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def pctile(s):
    """0..1 percentile of a series (higher = worse fragility for that input)."""
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True)
    return r


def main():
    # 1) leverage / coverage from fundamentals (latest per ticker)
    frag = pd.DataFrame()
    try:
        f = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
        if "as_of_date" in f.columns:
            f["as_of_date"] = pd.to_datetime(f["as_of_date"], errors="coerce")
            f = f.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
        frag["ticker"] = f["ticker"]
        for col in ("debt_to_equity", "mktcap_to_assets", "interest_coverage"):
            if col in f.columns:
                frag[f"{col}_raw"] = pd.to_numeric(f[col], errors="coerce")
        # interest coverage: lower = worse -> invert
        if "interest_coverage_raw" in frag.columns:
            frag["ic_inv_raw"] = 1.0 / frag["interest_coverage_raw"].replace(0, np.nan)
    except Exception as e:
        print("fundamentals:", e)

    # 2) IV skew from options_skew.csv (latest per ticker)
    skew = {}
    try:
        s = pd.read_csv(DATA_DIR / "options_skew.csv")
        s = s.sort_values("date") if "date" in s.columns else s
        for t, g in s.groupby("ticker"):
            skew[t] = float(g["skew"].iloc[-1])
    except Exception:
        pass
    if skew:
        sk = pd.DataFrame({"ticker": list(skew.keys()), "skew_raw": list(skew.values())})
        frag = frag.merge(sk, on="ticker", how="left")

    # 3) gap share + 4) tail alpha from the new Taleb scripts
    try:
        g = pd.read_csv(DATA_DIR / "gap_risk.csv")
        frag = frag.merge(g[["ticker", "gap_share_of_var", "p_abs_gap_gt_3pct"]], on="ticker", how="left")
    except Exception:
        pass
    try:
        t = pd.read_csv(DATA_DIR / "tail_index.csv")
        frag = frag.merge(t[["ticker", "tail_alpha_hill", "kurtosis"]], on="ticker", how="left")
    except Exception:
        pass

    # 5) illiquidity: avg daily dollar volume (last 2y of volume x close)
    cols = ["date", "ticker", "close", "volume"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    cutoff = pd.Timestamp.now().date().replace(year=pd.Timestamp.now().year - 2)
    d = d[d["date"].astype("datetime64[ns]").dt.date >= cutoff]
    d["dollar_vol"] = pd.to_numeric(d["close"], errors="coerce") * pd.to_numeric(d["volume"], errors="coerce")
    adv = d.groupby("ticker")["dollar_vol"].mean().rename("adv_usd")
    frag = frag.merge(adv.reset_index(), on="ticker", how="left")

    if frag.empty:
        print("no data")
        return

    # Build percentiles (higher = more fragile)
    comp = {}
    if "debt_to_equity_raw" in frag.columns:
        comp["leverage"] = pctile(frag["debt_to_equity_raw"])
    if "mktcap_to_assets_raw" in frag.columns:
        comp["asset_coverage"] = pctile(frag["mktcap_to_assets_raw"])
    if "ic_inv_raw" in frag.columns:
        comp["interest_coverage"] = pctile(frag["ic_inv_raw"])
    if "skew_raw" in frag.columns:
        comp["iv_skew"] = pctile(frag["skew_raw"])
    if "adv_usd" in frag.columns:
        comp["illiquidity"] = pctile(-frag["adv_usd"])  # low ADV = fragile
    if "gap_share_of_var" in frag.columns:
        comp["gap_share"] = pctile(frag["gap_share_of_var"])
    if "tail_alpha_hill" in frag.columns:
        comp["tail_fatness"] = pctile(-frag["tail_alpha_hill"])  # low alpha = fat = fragile
    if "kurtosis" in frag.columns:
        comp["kurtosis"] = pctile(frag["kurtosis"])

    # Taleb-style weights: leverage & gap & tail matter most
    W = {
        "leverage": 0.18, "asset_coverage": 0.12, "interest_coverage": 0.12,
        "iv_skew": 0.12, "illiquidity": 0.12, "gap_share": 0.14,
        "tail_fatness": 0.14, "kurtosis": 0.06,
    }
    total_w = sum(w for k, w in W.items() if k in comp)
    score = pd.Series(0.0, index=frag.index)
    for k, s in comp.items():
        if k in W:
            # missing component -> neutral 0.5 (median), not NaN-toxic
            score += W[k] * s.fillna(0.5)
    if total_w > 0:
        score = score / total_w

    out = frag[["ticker"]].copy()
    out["fragility_score"] = score.round(3)
    out["fragility_pctile"] = score.rank(pct=True).round(3)
    out["fragile_flag"] = out["fragility_pctile"] >= 0.90
    for k, s in comp.items():
        out[f"{k}_pct"] = s.round(3).to_numpy()
    # keep raw values for reference
    for c in ("debt_to_equity_raw", "mktcap_to_assets_raw", "skew_raw", "adv_usd", "gap_share_of_var", "tail_alpha_hill", "kurtosis"):
        if c in frag.columns:
            out[c] = frag[c]
    out = out.sort_values("fragility_score", ascending=False)
    out.to_csv(DATA_DIR / "fragility_screen.csv", index=False)

    n_fragile = int(out["fragile_flag"].sum())
    print(f"fragility_screen.csv: {len(out)} tickers | {n_fragile} flagged FRAGILE (top 10%)")
    print("\nMost fragile names (score, flag, drivers):")
    cols = ["ticker", "fragility_score", "fragility_pctile", "fragile_flag"] + \
           [f"{k}_pct" for k in comp if k in W]
    print(out[cols].head(10).to_string(index=False))
    print("\nLeast fragile:")
    print(out[cols].tail(5).to_string(index=False))
    print("\nNote: fragile_flag is a VETO input for inclusion gates — cheap + fragile = skip.")


if __name__ == "__main__":
    main()

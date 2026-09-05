"""adhoc_universe_pairs.py — ONE-SHOT: for each personal-book name (9 + ALL),
find the best mean-reverting pair partners across the WHOLE liquid universe,
using the repo's own engines (return-corr screen -> OLS on log prices ->
fixed-lag ADF residual -> OU half-life, exactly as pair_engine.py).

Writes adhoc_universe_pairs.parquet + prints the top candidates per name.
Short-leg pair candidates are POSITIVE-beta (long cheap leg / short rich leg);
negative-beta significant pairs are flagged separately (hedge-type, not pairs).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import sys

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR))
from pair_engine import fast_adf_residual, half_life
from analytics_common import liquid_listed_tickers

BOOK = ["BAYRY", "CAG", "HMC", "HPQ", "KHC", "MOS", "PFE", "SMCI", "T", "ALL"]
LOOK = 756          # OLS/ADF window (trading days, 3y)
CORR_N = 250        # return-corr screen window
CORR_MIN = 0.25     # screen gate: keep partners with corr >= this
MAX_PARTNERS = 250  # most-correlated partners to test per book name

liq = liquid_listed_tickers()
uni = sorted(liq | set(BOOK))
d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
tab = d.to_table(columns=["date", "ticker", "adj_close"],
                 filter=pc.field("ticker").isin(uni))
df = tab.to_pandas()
df["date"] = pd.to_datetime(df["date"]).dt.date
px = df.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
px = px[[c for c in uni if c in px.columns]]
cov = px.notna().sum()
keep = [t for t in uni if cov.get(t, 0) >= 500]
px = px[keep]
print(f"universe: {len(keep)} liquid names with >=500 obs, {len(px)} days")

rets = px.pct_change()
rows = []
for mk, book_tk in enumerate(BOOK):
    if book_tk not in px.columns:
        print(f"  {book_tk}: NOT in liquid universe — skip")
        continue
    # --- return-corr screen (vectorized over the universe) ---
    rc = rets[book_tk].tail(CORR_N)
    cs = rets[keep].tail(CORR_N).corrwith(rc)
    partners = cs.drop(labels=[book_tk], errors="ignore").sort_values(ascending=False)
    partners = partners[partners >= CORR_MIN].head(MAX_PARTNERS)
    if len(partners) == 0:
        print(f"  {book_tk}: no partner with corr>={CORR_MIN} in {CORR_N}d — none above screen")
        continue
    # --- OLS + ADF on log prices over LOOK window (shared clean window) ---
    sub = px[[book_tk] + partners.index.tolist()].dropna(how="any").iloc[-LOOK:]
    x0 = np.log(sub[book_tk].to_numpy(float))
    for ptk in partners.index:
        y = np.log(sub[ptk].to_numpy(float))
        ok = np.isfinite(x0) & np.isfinite(y)
        x, yy = x0[ok], y[ok]
        if len(x) < 250:
            continue
        xd, yd = x - x.mean(), yy - yy.mean()
        den = np.dot(xd, xd)
        if den <= 1e-18:
            continue
        beta = np.dot(xd, yd) / den
        resid = yy - (yy.mean() - beta * x.mean()) - beta * x
        t, p = fast_adf_residual(resid)
        hl = half_life(resid)
        c = float(cs[ptk])
        rows.append({"book_name": book_tk, "partner": ptk, "ret_corr": c,
                     "beta": beta, "adf_p": float(p), "half_life_d": hl})
    n_sig = sum(1 for r in rows if r["book_name"] == book_tk and r["adf_p"] < 0.05)
    print(f"  {book_tk}: {len(partners)} partners tested, {n_sig} with ADF p<0.05")

out = pd.DataFrame(rows)
out.to_parquet(DATA_DIR / "adhoc_universe_pairs.parquet", index=False)
print(f"\nwrote adhoc_universe_pairs.parquet ({len(out)} rows)")

if len(out):
    print("\n=== best POSITIVE-beta (classic long/short pair) candidates ===")
    pos = out[(out.beta > 0.3) & (out.adf_p < 0.05)].sort_values("adf_p")
    print(pos.head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}")
          if len(pos) else "(none above gates)")
    print("\n=== significant NEGATIVE-beta (hedge-type, NOT pairs) ===")
    neg = out[(out.beta < -0.2) & (out.adf_p < 0.05)].sort_values("adf_p")
    print(neg.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}")
          if len(neg) else "(none)")
else:
    print("no rows — everything failed the corr screen")

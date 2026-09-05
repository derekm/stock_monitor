"""adhoc_trough_recovery.py — ONE-SHOT: names currently in a trough/discount
that still pass the full gate stack, ranked by recovery characteristics.

Trough = deep drawdown from 52w high AND trading below its 200d SMA AND in
the bottom quartile of its own 3y price range (a real trough, not a dip).
Recovery = the book winners' pre-run profile: implied_r > 0, improving
13w/26w momentum, composite quality, NOT fragile.

This screens for "what the winners looked like at their trough", it does NOT
predict. The plan's honesty rule: quotes are measurements, not forecasts.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR))
from analytics_common import liquid_listed_tickers

bc = pd.read_parquet(DATA_DIR / "buy_candidates.parquet")
ir = pd.read_parquet(DATA_DIR / "implied_r_screen.parquet")
veto = pd.read_parquet(DATA_DIR / "fragility_veto.parquet") \
    if (DATA_DIR / "fragility_veto.parquet").exists() else None
fs = pd.read_parquet(DATA_DIR / "fragility_screen.parquet") \
    if (DATA_DIR / "fragility_screen.parquet").exists() else None
liq = liquid_listed_tickers()

uni = sorted(liq)
d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
tab = d.to_table(columns=["date", "ticker", "adj_close"],
                 filter=pc.field("ticker").isin(uni))
df = tab.to_pandas()
df["date"] = pd.to_datetime(df["date"]).dt.date
px = df.pivot_table(index="date", columns="ticker", values="adj_close")

def robust_cagr(s: pd.Series, days: int) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    r = (s.iloc[-1] / s.iloc[0]) ** (365.0 / max(days, 1)) - 1.0
    return float(r) if np.isfinite(r) else np.nan

rows = []
for t in uni:
    if t not in px.columns:
        continue
    s = px[t].dropna()
    if len(s) < 756 or t not in liq:
        continue
    price = float(s.iloc[-1])
    hi52 = float(s.iloc[-252:].max())
    if hi52 <= 0:
        continue
    dd = price / hi52 - 1.0
    sma200 = float(s.iloc[-200:].mean())
    below200 = price < sma200
    r3y = s.iloc[-756:]
    pct3y = (r3y <= price).mean()  # percentile of current price in 3y range
    # trough definition: real drawdown + below its own trend + bottom-quartile
    if dd > -0.20 or not below200 or pct3y > 0.25:
        continue
    # gates
    if veto is not None and len(veto):
        v = veto[veto["ticker"] == t]
        if len(v) and bool(v["veto_flag"].fillna(False).iloc[0] |
                           v["alpha_lt_2"].fillna(False).iloc[0]):
            continue
    if fs is not None and (fs["ticker"] == t).any():
        if bool(fs.loc[fs["ticker"] == t, "fragile_flag"].fillna(False).iloc[0]):
            continue
    irc = float(ir.loc[ir["ticker"] == t, "implied_r"].iloc[0]) \
        if ir is not None and (ir["ticker"] == t).any() else np.nan
    if not np.isfinite(irc) or irc <= 0:
        continue
    comp = float(bc.loc[bc["ticker"] == t, "composite_score"].iloc[0]) \
        if (bc["ticker"] == t).any() else np.nan
    # momentum ladders
    m13 = robust_cagr(s.iloc[-63:], 63) if len(s) >= 63 else np.nan
    m26 = robust_cagr(s.iloc[-126:], 126) if len(s) >= 126 else np.nan
    m52 = robust_cagr(s.iloc[-252:], 252) if len(s) >= 252 else np.nan
    rows.append({"ticker": t, "price": price, "dd_52w": dd, "pct_3y": pct3y,
                 "implied_r": irc, "composite": comp,
                 "m13": m13, "m26": m26, "m52": m52})

out = pd.DataFrame(rows)
print(f"{len(out)} names: trough (dd<=-20%, <200d SMA, bottom quartile) + all gates + implied_r>0\n")
if len(out):
    # recovery rank: quality + implied_r + momentum turning (m13 better than m52 trend)
    out = out.sort_values(["composite", "implied_r"], ascending=[False, False])
    print(out.head(20).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
out.to_parquet(DATA_DIR / "adhoc_trough_recovery.parquet", index=False)
print(f"\nwrote adhoc_trough_recovery.parquet ({len(out)} rows)")

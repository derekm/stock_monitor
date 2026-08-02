#!/usr/bin/env python3
"""Add realized vol, beta, max DD to preferred_metrics and fundamentals analytics."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent
PRICES = DATA / "daily_prices.parquet"
PREF = DATA / "preferred_metrics.csv"
PREF_PQ = DATA / "preferred_metrics.parquet"
FUND = DATA / "fundamentals.parquet"


def metrics_for(tickers, rets, mkt):
    out = {}
    for t in tickers:
        if t not in rets.columns:
            out[t] = dict(name_vol=np.nan, beta=np.nan, max_dd=np.nan)
            continue
        r = rets[t].dropna()
        vol = float(r.iloc[-63:].std() * np.sqrt(252)) if len(r) >= 20 else float("nan")
        aligned = pd.concat([r, mkt], axis=1, keys=["a", "m"]).dropna().iloc[-126:]
        beta = float(aligned.cov().iloc[0, 1] / aligned["m"].var()) if len(aligned) > 20 and aligned["m"].var() > 0 else np.nan
        cum = r.cumsum()
        max_dd = float((np.exp(cum) / np.exp(cum).cummax() - 1).min()) if len(r) > 5 else np.nan
        out[t] = dict(name_vol=vol, beta=beta, max_dd=max_dd)
    return out


def main():
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)

    pref = pd.read_csv(PREF)
    m = metrics_for(pref["ticker"].tolist(), rets, mkt)
    for col in ("name_vol", "beta", "max_dd"):
        pref[col] = pref["ticker"].map(lambda t: m.get(t, {}).get(col))
    pref.to_csv(PREF, index=False)
    try:
        pref.to_parquet(PREF_PQ, index=False)
    except Exception:
        pass
    print("preferred_metrics enriched with name_vol, beta, max_dd")
    print(pref[["ticker", "decision", "composite_score", "name_vol", "beta", "max_dd"]].head(10).to_string(index=False))

    # also write risk_metrics.csv standalone
    rows = [{"ticker": t, **v} for t, v in m.items()]
    pd.DataFrame(rows).to_csv(DATA / "risk_metrics.csv", index=False)
    print("Wrote risk_metrics.csv")


if __name__ == "__main__":
    main()

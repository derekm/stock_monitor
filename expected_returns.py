#!/usr/bin/env python3
"""
expected_returns.py — Ilmanen 4-pillar expected-return scores.

Pillars (cross-sectional percentile ranks, higher = more attractive):
  carry      earnings yield + FCF yield (NI_ttm / mcap, FCF / mcap)
  value      B/M + E/P + FCF yield + S/P
  momentum   12-1 price momentum (252d return, skip last 21d)
  defensive  low 60d vol + low |beta| + quality (ROE, ROIC, −D/E)

Composite expected_return is the equal-weight mean of available pillars.

Output (month-end, long): expected_returns_decomp.parquet
  date, ticker, carry, value, momentum, defensive, expected_return
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent


def _snapshot(src: Path) -> Path:
    """Copy parquet before read so a writer can still os.replace the live file."""
    tmp = Path(tempfile.gettempdir()) / f"er_{src.name}"
    shutil.copy2(src, tmp)
    return tmp


def _to_date_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(idx), errors="coerce"))


def _pivot_fund(fund: pd.DataFrame, col: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if col not in fund.columns:
        return pd.DataFrame(index=calendar)
    d = fund.dropna(subset=["date", "ticker", col]).copy()
    d["ticker"] = d["ticker"].astype(str).str.upper()
    d = d.drop_duplicates(subset=["date", "ticker"], keep="last")
    piv = d.pivot(index="date", columns="ticker", values=col)
    piv.index = _to_date_index(piv.index)
    piv = piv.sort_index().ffill().reindex(calendar).ffill()
    return piv


def _rank(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)


def _mean_legs(legs: list[pd.DataFrame]) -> pd.DataFrame:
    if not legs:
        return pd.DataFrame()
    idx = legs[0].index
    cols = legs[0].columns
    for g in legs[1:]:
        idx = idx.union(g.index)
        cols = cols.union(g.columns)
    acc = None
    cnt = None
    for g in legs:
        a = g.reindex(index=idx, columns=cols)
        acc = a.fillna(0) if acc is None else acc.add(a.fillna(0))
        present = a.notna().astype("float64")
        cnt = present if cnt is None else cnt.add(present)
    return acc.div(cnt.replace(0, np.nan))


def load_close_mcap() -> tuple[pd.DataFrame, pd.DataFrame]:
    src = _snapshot(DATA_DIR / "daily_prices.parquet")
    cols = ["date", "ticker", "adj_close", "close", "market_cap"]
    prices = pd.read_parquet(src, columns=cols)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.drop_duplicates(subset=["date", "ticker"], keep="last")
    px = prices["adj_close"].where(prices["adj_close"].notna(), prices["close"])
    prices = prices.assign(px=px)
    close = prices.pivot(index="date", columns="ticker", values="px")
    mcap = prices.pivot(index="date", columns="ticker", values="market_cap")
    close.index = _to_date_index(close.index)
    mcap.index = _to_date_index(mcap.index)
    close = close.sort_index().ffill(limit=5)
    mcap = mcap.sort_index().ffill()
    return close, mcap


def mcap_from_fund(fund: pd.DataFrame, close: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    """Prefer PIT daily_mcap.parquet; else shares × close; else fund market_cap."""
    out = mcap.reindex(index=close.index, columns=close.columns)
    panel_path = DATA_DIR / "daily_mcap.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        panel["ticker"] = panel["ticker"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        wide = panel.pivot(index="date", columns="ticker", values="market_cap")
        idx = pd.to_datetime(close.index).normalize()
        wide = wide.reindex(index=idx, columns=close.columns)
        wide.index = close.index
        out = wide.where(wide.notna() & (wide > 0), out)
    shares = _pivot_fund(fund, "shares_outstanding", close.index)
    if not shares.empty:
        sh = shares.reindex(index=close.index, columns=close.columns)
        implied = sh * close
        out = out.where(out.notna() & (out > 0), implied)
    fund_m = _pivot_fund(fund, "market_cap", close.index)
    if not fund_m.empty:
        fm = fund_m.reindex(index=close.index, columns=close.columns)
        out = out.where(out.notna() & (out > 0), fm)
    return out.replace(0, np.nan)


def load_fundamentals() -> pd.DataFrame:
    src = _snapshot(DATA_DIR / "fundamentals.parquet")
    fund = pd.read_parquet(src)
    date_col = "as_of_date" if "as_of_date" in fund.columns else "date"
    fund = fund.rename(columns={date_col: "date"})
    fund["ticker"] = fund["ticker"].astype(str).str.upper()
    return fund


def compute_carry(fund: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    """Equity carry: earnings yield + FCF yield, ranked."""
    ni = _pivot_fund(fund, "net_income_ttm", mcap.index)
    fcf = _pivot_fund(fund, "free_cash_flow", mcap.index)
    m = mcap.replace(0, np.nan)
    legs = []
    if not ni.empty:
        legs.append(_rank(ni.reindex(index=m.index, columns=m.columns).div(m)))
    if not fcf.empty:
        legs.append(_rank(fcf.reindex(index=m.index, columns=m.columns).div(m)))
    return _mean_legs(legs)


def compute_value(fund: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    """Value: B/M, E/P, FCF yield, S/P, ranked then averaged."""
    m = mcap.replace(0, np.nan)
    specs = [
        ("shareholders_equity", False),
        ("net_income_ttm", False),
        ("free_cash_flow", False),
        ("revenue_ttm", False),
    ]
    legs = []
    for col, _ in specs:
        piv = _pivot_fund(fund, col, m.index)
        if piv.empty:
            continue
        legs.append(_rank(piv.reindex(index=m.index, columns=m.columns).div(m)))
    return _mean_legs(legs)


def compute_momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum (Jegadeesh/Titman)."""
    mom = close.pct_change(lookback).shift(skip)
    return _rank(mom)


def compute_defensive(close: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Low vol, low |beta|, high quality."""
    rets = close.pct_change()
    vol = rets.rolling(60, min_periods=40).std() * np.sqrt(252)
    low_vol = _rank(-vol)

    mkt = rets.mean(axis=1)
    cov = rets.rolling(252, min_periods=126).cov(mkt)
    var = mkt.rolling(252, min_periods=126).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)
    low_beta = _rank(-beta.abs())

    legs = [low_vol, low_beta]
    for col, invert in (("roe", False), ("roic", False), ("debt_to_equity", True)):
        piv = _pivot_fund(fund, col, close.index)
        if piv.empty:
            continue
        aligned = piv.reindex(index=close.index, columns=close.columns)
        legs.append(_rank(-aligned if invert else aligned))
    return _mean_legs(legs)


def month_end_long(pillars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack month-end ranks to long ticker×date."""
    sample = next(iter(pillars.values()))
    me = sample.resample("ME").last().index
    frames = {}
    for name, wide in pillars.items():
        if wide is None or wide.empty:
            continue
        frames[name] = wide.reindex(me).ffill()
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "expected_return"])
    stacked = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    stacked["n_pillars"] = stacked[["carry", "value", "momentum", "defensive"]].notna().sum(axis=1)
    stacked["expected_return"] = stacked[["carry", "value", "momentum", "defensive"]].mean(axis=1, skipna=True)
    stacked.loc[stacked["n_pillars"] < 2, "expected_return"] = np.nan
    out = stacked.reset_index()
    out.columns = ["date", "ticker", *stacked.columns]
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return apply_er_eligibility(out)


def apply_er_eligibility(out: pd.DataFrame) -> pd.DataFrame:
    """ER is defined only for common stock with at least one of carry/value."""
    out = out.copy()
    stocks = DATA_DIR / "monitored_stocks.parquet"
    itype = pd.Series("stock", index=out["ticker"].unique())
    if stocks.exists():
        ms = pd.read_parquet(stocks, columns=["ticker", "instrument_type"])
        ms["ticker"] = ms["ticker"].astype(str).str.upper()
        itype = ms.drop_duplicates("ticker").set_index("ticker")["instrument_type"]
    out["instrument_type"] = out["ticker"].map(itype).fillna("stock")
    legs = [c for c in ["carry", "value", "momentum", "defensive"] if c in out.columns]
    out["n_pillars"] = out[legs].notna().sum(axis=1)
    out["expected_return"] = out[legs].mean(axis=1, skipna=True)
    fund = out["carry"].notna() | out["value"].notna() if "carry" in out.columns else False
    ok = (out["instrument_type"] == "stock") & fund & (out["n_pillars"] >= 2)
    out.loc[~ok, "expected_return"] = np.nan
    return out


def oos_direction() -> pd.DataFrame:
    """Top-quintile ER vs EW, next calendar month, eligible names only."""
    import tempfile
    er = pd.read_parquet(DATA_DIR / "expected_returns_decomp.parquet")
    er = er.dropna(subset=["expected_return"])
    er["date"] = pd.to_datetime(er["date"])
    snap = Path(tempfile.gettempdir()) / "ph_daily_prices.parquet"
    px = pd.read_parquet(snap, columns=["date", "ticker", "adj_close", "close"])
    px["ticker"] = px["ticker"].astype(str).str.upper()
    keep = set(er["ticker"].unique())
    px = px[px["ticker"].isin(keep)]
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["px"] = px["adj_close"].where(px["adj_close"].notna(), px["close"])
    last = px.sort_values("date").groupby(["ticker", px["date"].dt.to_period("M")]).tail(1)
    last["month"] = last["date"].dt.to_period("M")
    wide = last.pivot(index="month", columns="ticker", values="px")
    fwd = wide.shift(-1) / wide - 1.0
    rows = []
    for dt, g in er.groupby(er["date"].dt.to_period("M")):
        if dt not in fwd.index:
            continue
        r = fwd.loc[dt]
        g = g.drop_duplicates("ticker").set_index("ticker")
        both = g.index.intersection(r.dropna().index)
        if len(both) < 50:
            continue
        sc = g.loc[both, "expected_return"]
        rr = r.loc[both]
        rr = rr[rr.abs().le(0.50)]
        both = rr.index.intersection(sc.index)
        if len(both) < 50:
            continue
        sc = sc.loc[both]
        rr = rr.loc[both]
        q = sc.quantile(0.80)
        top = rr[sc >= q]
        rows.append({
            "month": str(dt),
            "n": int(len(both)),
            "ew": float(rr.mean()),
            "top": float(top.mean()),
            "hit_top": float((top > 0).mean()),
            "hit_ew": float((rr > 0).mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        print("OOS: no overlapping months")
        return out
    edge = out["top"].mean() - out["ew"].mean()
    hit_edge = out["hit_top"].mean() - out["hit_ew"].mean()
    print(out.tail(8).round(4).to_string(index=False))
    print(f"months {len(out)}  top-EW {edge:.3%}  hit-edge {hit_edge:.1%}  "
          f"(gate +5% ret or +5pp hit)")
    out.to_parquet(DATA_DIR / "er_oos_direction.parquet", index=False)
    return out


def main() -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--oos", action="store_true",
                    help="Month-end ER top-quintile vs EW next-month return (no price write)")
    args = ap.parse_args()

    if args.oos:
        return oos_direction()

    print("Loading prices (snapshot)...")
    close, mcap_px = load_close_mcap()
    print(f"  {close.shape[0]} dates × {close.shape[1]} tickers")

    print("Loading fundamentals (snapshot)...")
    fund = load_fundamentals()
    mcap = mcap_from_fund(fund, close, mcap_px)
    print(f"  mcap coverage last: {int(mcap.iloc[-1].notna().sum()):,} / {mcap.shape[1]}")

    print("Computing pillars...")
    print("  carry (EY + FCF yield)")
    carry = compute_carry(fund, mcap)
    print("  value (B/M E/P FCFY S/P)")
    value = compute_value(fund, mcap)
    print("  momentum (12-1)")
    momentum = compute_momentum(close, args.lookback)
    print("  defensive (vol/beta/quality)")
    defensive = compute_defensive(close, fund)

    out = month_end_long({
        "carry": carry,
        "value": value,
        "momentum": momentum,
        "defensive": defensive,
    })

    if args.save:
        path = DATA_DIR / "expected_returns_decomp.parquet"
        out.to_parquet(path, index=False)
        print(f"\nSaved {path} ({len(out):,} rows)")

    print("\n=== Coverage (month-end, last date) ===")
    if not out.empty:
        last = out["date"].max()
        latest = out[out["date"] == last]
        print(f"  as-of {last}  n={len(latest):,}")
        for col in [c for c in latest.columns if c not in ("date", "ticker")]:
            n = int(latest[col].notna().sum())
            print(f"  {col}: {n:,} / {len(latest):,} ({n / max(len(latest), 1) * 100:.1f}%)")
        shown = latest.dropna(subset=["expected_return"]).nlargest(10, "expected_return")
        print(f"\n=== Top 10 ER @ {last} (stock + carry|value) ===")
        cols = [c for c in ["ticker", "n_pillars", "carry", "value", "momentum", "defensive", "expected_return"] if c in shown]
        print(shown[cols].to_string(index=False))
    return out


if __name__ == "__main__":
    main()

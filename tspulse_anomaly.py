#!/usr/bin/env python3
"""
tspulse_anomaly.py — Anomaly detection for stock series (TSPulse-ready).

IBM Granite TSPulse targets time-series anomaly detection / classification.
This module:
  1. Tries to load a TSPulse / Granite anomaly model when available
  2. Falls back to robust statistical detectors that run offline:
       - Rolling z-score on returns
       - Residual z-score vs local median
       - Volume spike detector
       - Multivariate dispersion shock (market-wide)

Usage:
  python tspulse_anomaly.py scan --ticker MOS,CF,SHEL
  python tspulse_anomaly.py scan --index portfolio --z 3.0
  python tspulse_anomaly.py scan --index fertilizer --save
  python tspulse_anomaly.py status
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
ANOMALY_FILE = DATA_DIR / "anomalies_tspulse.csv"
EXOG_FILE = DATA_DIR / "exogenous_panel.parquet"

# Candidate HF ids (may change as IBM publishes)
TSPULSE_MODELS = [
    "ibm-granite/granite-timeseries-tspulse",
    "ibm-granite/TSPulse",
]


def load_prices(tickers: Optional[list[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    return df.sort_values(["ticker", "date"])


def resolve_tickers_from_args(args, default_index='portfolio') -> list[str]:
    if getattr(args, "ticker", None):
        return [x.strip().upper() for x in args.ticker.split(",") if x.strip()]
    if getattr(args, "index", None):
        try:
            names = parse_indexes(args.index)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        seen, out = set(), []
        for n in names:
            for tk in tickers_for_index(n):
                if tk not in seen:
                    seen.add(tk)
                    out.append(tk)
        return out
    # Default: FULL universe (all tickers with prices), not just portfolio/fertilizer
    df = pd.read_parquet(PRICES_FILE, columns=["ticker"])
    return sorted(df["ticker"].astype(str).str.upper().unique())



def try_load_tspulse(model_name: Optional[str] = None):
    """Attempt TSPulse / Granite anomaly model load."""
    names = [model_name] if model_name else TSPULSE_MODELS
    for name in names:
        if not name:
            continue
        try:
            from transformers import AutoModel
            import torch  # noqa: F401
            print(f"Loading TSPulse-like model: {name}")
            model = AutoModel.from_pretrained(name, trust_remote_code=True)
            model.eval()
            return model, name
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {e}")
    return None, None


def statistical_anomalies(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    z_thresh: float = 3.0,
    window: int = 20,
) -> pd.DataFrame:
    """
    Offline anomaly scores:
      ret_z      — rolling z of log returns
      resid_z    — z of residual vs rolling median
      vol_z      — volume z-score
      is_anomaly — any |score| >= z_thresh
    """
    close = close.dropna().sort_index()
    ret = np.log(close / close.shift(1))
    mu = ret.rolling(window, min_periods=max(5, window // 2)).mean()
    sd = ret.rolling(window, min_periods=max(5, window // 2)).std().replace(0, np.nan)
    ret_z = (ret - mu) / sd

    med = close.rolling(window, min_periods=max(5, window // 2)).median()
    resid = close - med
    rsd = resid.rolling(window, min_periods=max(5, window // 2)).std().replace(0, np.nan)
    resid_z = resid / rsd

    out = pd.DataFrame({
        "close": close,
        "ret": ret,
        "ret_z": ret_z,
        "resid_z": resid_z,
    })
    if volume is not None:
        v = volume.reindex(close.index).fillna(0)
        vmu = v.rolling(window, min_periods=5).mean()
        vsd = v.rolling(window, min_periods=5).std().replace(0, np.nan)
        out["vol_z"] = (v - vmu) / vsd
    else:
        out["vol_z"] = np.nan

    out["score"] = out[["ret_z", "resid_z"]].abs().max(axis=1)
    if out["vol_z"].notna().any():
        out["score"] = out[["score", "vol_z"]].abs().max(axis=1)
    out["is_anomaly"] = out["score"] >= z_thresh
    out["backend"] = "statistical"
    return out


def tspulse_anomalies(model, close: pd.Series, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Hook for real TSPulse inference. Until a stable public API is wired,
    falls through to statistical with backend tag if model path fails.
    """
    try:
        import torch
        y = close.dropna().values.astype(np.float32)
        x = torch.tensor(y).view(1, -1, 1)
        with torch.no_grad():
            out = model(x)
            # Heuristic: treat reconstruction error / anomaly head if present
            if hasattr(out, "anomaly_scores"):
                scores = out.anomaly_scores.detach().cpu().numpy().reshape(-1)
            elif hasattr(out, "logits"):
                scores = out.logits.detach().cpu().numpy().reshape(-1)
            else:
                rec = out[0] if isinstance(out, (tuple, list)) else out
                rec = rec.detach().cpu().numpy().reshape(-1)
                n = min(len(rec), len(y))
                scores = np.abs(y[-n:] - rec[-n:])
                scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        idx = close.dropna().index[-len(scores):]
        df = pd.DataFrame({"score": scores, "is_anomaly": np.abs(scores) >= z_thresh}, index=idx)
        df["close"] = close.reindex(idx)
        df["backend"] = "tspulse"
        return df
    except Exception as e:
        print(f"  TSPulse inference failed ({e}); statistical fallback")
        return statistical_anomalies(close, z_thresh=z_thresh)


def scan_ticker(
    ticker: str,
    prices: pd.DataFrame,
    model=None,
    z_thresh: float = 3.0,
    window: int = 20,
) -> pd.DataFrame:
    sub = prices[prices["ticker"] == ticker].set_index("date").sort_index()
    if sub.empty or "close" not in sub.columns:
        return pd.DataFrame()
    close = sub["close"].dropna()
    vol = sub["volume"] if "volume" in sub.columns else None
    if model is not None:
        res = tspulse_anomalies(model, close, z_thresh=z_thresh)
        # merge volume z from statistical for context
        stat = statistical_anomalies(close, vol, z_thresh=z_thresh, window=window)
        for c in ["ret_z", "resid_z", "vol_z", "ret"]:
            if c in stat.columns and c not in res.columns:
                res[c] = stat[c]
        return res
    return statistical_anomalies(close, vol, z_thresh=z_thresh, window=window)


def cmd_status(args):
    print("TSPulse anomaly module")
    print(f"  prices: {PRICES_FILE.exists()}")
    model, name = try_load_tspulse(getattr(args, "model", None))
    print(f"  model : {name or 'none (statistical backend)'}")
    if EXOG_FILE.exists():
        print(f"  exog  : {EXOG_FILE}")


def cmd_scan(args):
    tickers = resolve_tickers_from_args(args, default_index='portfolio')
    prices = load_prices(tickers)
    model, name = try_load_tspulse(args.model)
    z = args.z
    rows = []
    print(f"Scanning {len(tickers)} tickers  z={z}  backend={'tspulse' if model else 'statistical'}")

    # Market-wide dispersion shock from exog if present
    mkt_flags = {}
    if EXOG_FILE.exists():
        exog = pd.read_parquet(EXOG_FILE)
        if "date" in exog.columns:
            exog["date"] = pd.to_datetime(exog["date"])
            exog = exog.set_index("date")
        else:
            exog.index = pd.to_datetime(exog.index)
        if "dispersion" in exog.columns:
            d = exog["dispersion"]
            dz = (d - d.rolling(20).mean()) / d.rolling(20).std()
            mkt_flags = dz[dz.abs() >= z].to_dict()

    for t in tickers:
        res = scan_ticker(t, prices, model=model, z_thresh=z, window=args.window)
        if res.empty:
            continue
        anom = res[res["is_anomaly"] == True]
        print(f"\n{t}: {len(anom)} anomalies / {len(res)} days  "
              f"(max |score|={res['score'].abs().max():.2f})")
        if len(anom):
            show = anom.tail(min(5, len(anom)))
            for dt, r in show.iterrows():
                print(f"  {pd.Timestamp(dt).date()}  close={r.get('close', float('nan')):.2f}  "
                      f"score={r['score']:.2f}  ret_z={r.get('ret_z', float('nan')):.2f}")
        for dt, r in anom.iterrows():
            rows.append({
                "ticker": t,
                "date": pd.Timestamp(dt),
                "close": r.get("close"),
                "score": round(float(r["score"]), 4),
                "ret_z": round(float(r["ret_z"]), 4) if pd.notna(r.get("ret_z")) else np.nan,
                "resid_z": round(float(r["resid_z"]), 4) if pd.notna(r.get("resid_z")) else np.nan,
                "vol_z": round(float(r["vol_z"]), 4) if pd.notna(r.get("vol_z")) else np.nan,
                "backend": r.get("backend", "statistical"),
                "mkt_dispersion_shock": bool(pd.Timestamp(dt) in mkt_flags),
            })

    if mkt_flags:
        print(f"\nMarket dispersion shocks (|z|>={z}): {len(mkt_flags)} days")

    if rows and args.save:
        pd.DataFrame(rows).to_csv(ANOMALY_FILE, index=False)
        print(f"\nWrote {ANOMALY_FILE} ({len(rows)} events)")
    elif rows:
        print(f"\n{len(rows)} anomaly events (pass --save to write CSV)")


def main():
    parser = argparse.ArgumentParser(description="TSPulse / statistical anomaly detection")
    parser.add_argument("--model", default=None, help="HF model id override")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("scan")
    p.add_argument("--ticker")
    p.add_argument("--index", help=index_help_text())
    p.add_argument("--z", type=float, default=3.0, help="|z| threshold")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

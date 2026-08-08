#!/usr/bin/env python3
"""
analyze_granite_forecasts.py — Summarize Granite forecasts and compare tickers.

Usage:
  python analyze_granite_forecasts.py
  python analyze_granite_forecasts.py --ticker MOS,CF,SHEL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
FORECAST_CSV = DATA_DIR / "forecasts_granite.csv"
FORECAST_PQ = DATA_DIR / "forecasts_granite.parquet"
BACKTEST_FILE = DATA_DIR / "forecast_backtest_metrics.csv"
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
REGIME_STATS = DATA_DIR / "regime_forecast_stats.csv"
HMM_FILE = DATA_DIR / "hmm_regime_states.csv"
REGIME_BEST = DATA_DIR / "regime_model_best.csv"


def load_regime_selection() -> tuple[str | None, dict[str, dict]]:
    """(current regime, {ticker: {steps, cap, lr, dir_acc, pers_dir, excess}}).

    Regime-SELECTED models: pass6 trained one model per HMM regime and picked
    the best config per (ticker, regime) by max OOS direction excess over the
    regime's persistence baseline. This loads that table and returns the
    config that SHOULD be used for the CURRENT regime — the production
    consumption of pass6. Falls back to ({}, None) when unavailable.
    """
    regime_now = None
    if HMM_FILE.exists():
        try:
            hmm = pd.read_csv(HMM_FILE)
            if "date" in hmm.columns and "regime" in hmm.columns:
                hmm["date"] = pd.to_datetime(hmm["date"], errors="coerce")
                hmm = hmm.dropna(subset=["date"]).sort_values("date")
                if len(hmm):
                    regime_now = str(hmm.iloc[-1]["regime"])
        except Exception:
            pass
    sel: dict[str, dict] = {}
    if REGIME_BEST.exists():
        try:
            rb = pd.read_csv(REGIME_BEST)
            for _, r in rb.iterrows():
                tk = str(r.get("ticker", "")).upper()
                reg = str(r.get("regime", ""))
                if not tk or reg != regime_now:
                    continue
                sel[tk] = {
                    "steps": int(r.get("steps", 0)) if pd.notna(r.get("steps")) else None,
                    "cap": int(r.get("cap", 0)) if pd.notna(r.get("cap")) else None,
                    "lr": r.get("lr"),
                    "dir_acc": float(r.get("dir_acc")) if pd.notna(r.get("dir_acc")) else None,
                    "pers_dir": float(r.get("pers_dir")) if pd.notna(r.get("pers_dir")) else None,
                    "excess": float(r.get("dir_acc", 0) - r.get("pers_dir", 50))
                              if pd.notna(r.get("dir_acc")) and pd.notna(r.get("pers_dir")) else None,
                }
                for s in (10, 21, 42, 63, 96):
                    col = f"dir_acc_h{s}"
                    if col in rb.columns and pd.notna(r.get(col)):
                        sel[tk][f"dir_acc_h{s}"] = float(r[col])
        except Exception:
            pass
    return regime_now, sel


def load_regime_gate() -> tuple[str | None, dict[str, float]]:
    """(current regime, {ticker: per-regime persistence dir-acc baseline}).

    Returns (None, {}) when regime data is unavailable — the caller then
    falls back to the raw signal with no gating. The persistence baseline is
    the fraction of up-windows in the model's OOS test set per regime; a
    forecast should be read against it, not against 50%.
    """
    regime_now = None
    if HMM_FILE.exists():
        try:
            hmm = pd.read_csv(HMM_FILE)
            if "date" in hmm.columns and "regime" in hmm.columns:
                hmm["date"] = pd.to_datetime(hmm["date"], errors="coerce")
                hmm = hmm.dropna(subset=["date"]).sort_values("date")
                if len(hmm):
                    regime_now = str(hmm.iloc[-1]["regime"])
        except Exception:
            pass
    baselines: dict[str, float] = {}
    if REGIME_STATS.exists():
        try:
            rs = pd.read_csv(REGIME_STATS)
            col = f"persistence_dir_acc_by_regime"
            for _, r in rs.iterrows():
                tk = str(r.get("ticker", "")).upper()
                raw = r.get(col)
                if not tk or pd.isna(raw):
                    continue
                try:
                    d = eval(raw) if isinstance(raw, str) else dict(raw or {})
                except Exception:
                    continue
                if regime_now and regime_now in d and d[regime_now] is not None:
                    baselines[tk] = float(d[regime_now]) / 100.0
        except Exception:
            pass
    return regime_now, baselines


def load_forecasts() -> pd.DataFrame:
    if FORECAST_PQ.exists():
        try:
            return pd.read_parquet(FORECAST_PQ)
        except Exception:
            pass
    if FORECAST_CSV.exists():
        return pd.read_csv(FORECAST_CSV, parse_dates=["as_of", "forecast_date"])
    raise SystemExit("No forecasts found. Run: python forecast_granite.py forecast --ticker MOS")


def main():
    parser = argparse.ArgumentParser(description="Analyze Granite stock forecasts")
    parser.add_argument("--ticker", help="Filter tickers (comma-separated)")
    parser.add_argument("--index", help="Filter by index_name (comma-separated; substring match per label)")
    args = parser.parse_args()

    fc = load_forecasts()
    if args.ticker:
        tickers = [x.strip().upper() for x in args.ticker.split(",")]
        fc = fc[fc["ticker"].isin(tickers)]
    if args.index and "index_name" in fc.columns:
        wanted = [x.strip().lower() for x in args.index.split(",")]
        def _match(cell):
            labels = [p.strip().lower() for p in str(cell).split(",") if p.strip()]
            return any(w in labels for w in wanted)
        fc = fc[fc["index_name"].map(_match)]

    print("=" * 70)
    print("GRANITE FORECAST SUMMARY")
    print("=" * 70)
    print(f"As-of dates: {fc['as_of'].min()} → {fc['as_of'].max()}")
    print(f"Tickers: {sorted(fc['ticker'].unique())}")
    if "index_name" in fc.columns:
        print(f"Indexes: {sorted({p.strip() for s in fc['index_name'].dropna().astype(str) for p in s.split(',') if p.strip()})}")
    print(f"Horizons: {sorted(fc['horizon'].unique())}")
    print(f"Backend: {fc['backend'].iloc[0] if len(fc) else 'n/a'}")

    # Point forecast table at max horizon
    max_h = fc["horizon"].max()
    tail = fc[fc["horizon"] == max_h].copy()
    tail["signal"] = tail["pct_change"].apply(
        lambda x: "BULL" if x > 3 else ("BEAR" if x < -3 else "NEUTRAL")
    )

    # Regime gate: annotate each forecast against the per-regime persistence
    # baseline (fraction of up-windows the market realized in this regime).
    regime_now, baselines = load_regime_gate()
    regime_now_s, selection = load_regime_selection()
    if regime_now_s:
        regime_now = regime_now_s
    if regime_now:
        tail["regime"] = regime_now
        tail["pers_baseline"] = tail["ticker"].map(baselines)
        def _gate(row):
            if pd.isna(row.get("pers_baseline")):
                return row["signal"]
            p = float(row["pers_baseline"])
            # signal is meaningful only if the model's direction edge exceeds
            # what persistence already predicts in this regime
            if row["signal"] == "BULL" and p >= 0.60:
                return "BULL*"
            if row["signal"] == "BEAR" and p <= 0.40:
                return "BEAR*"
            return row["signal"]
        tail["signal_gated"] = tail.apply(_gate, axis=1)
        print(f"\nRegime gate: {regime_now} (per-regime persistence baseline; * = edge over baseline)")
        if selection:
            tail["regime_model_dir"] = tail["ticker"].map(
                lambda t: selection[t]["dir_acc"] if t in selection else None)
            tail["regime_model_excess"] = tail["ticker"].map(
                lambda t: selection[t]["excess"] if t in selection else None)
            for s in (10, 21, 42, 63, 96):
                col = f"regime_model_dir_h{s}"
                tail[col] = tail["ticker"].map(
                    lambda t, s=s: selection[t].get(f"dir_acc_h{s}") if t in selection else None)
            n_sel = tail["regime_model_dir"].notna().sum()
            print(f"Regime-selected models (pass6): {n_sel} tickers have a {regime_now} "
                  f"model; mean OOS dir {tail['regime_model_dir'].mean():.1f}% "
                  f"(excess +{tail['regime_model_excess'].mean():.1f}pt)")
    print(f"\n--- Horizon H+{max_h} snapshot ---")
    cols = ["ticker", "last_close", "forecast_close", "pct_change", "signal"]
    if "signal_gated" in tail.columns:
        cols.append("signal_gated")
        cols.append("pers_baseline")
    print(tail[cols].sort_values("pct_change", ascending=False).to_string(index=False))

    # Path shape: early vs late horizon
    print("\n--- Term structure (mean % change by horizon) ---")
    ts = fc.groupby("horizon")["pct_change"].mean()
    for h, v in ts.items():
        bar = "+" * max(0, int(v)) + "-" * max(0, int(-v))
        print(f"  H+{int(h):02d}  {v:+6.2f}%  {bar}")

    # Per-ticker expected move
    print("\n--- Expected move by ticker (final horizon) ---")
    for _, r in tail.sort_values("pct_change", ascending=False).iterrows():
        print(f"  {r['ticker']:6}  {r['pct_change']:+6.2f}%  ({r['last_close']:.2f} → {r['forecast_close']:.2f})")

    if BACKTEST_FILE.exists():
        bt = pd.read_csv(BACKTEST_FILE)
        if args.ticker:
            bt = bt[bt["ticker"].isin([x.strip().upper() for x in args.ticker.split(",")])]
        if args.index and "index_name" in bt.columns:
            wanted = [x.strip().lower() for x in args.index.split(",")]
            def _match(cell):
                labels = [p.strip().lower() for p in str(cell).split(",") if p.strip()]
                return any(w in labels for w in wanted)
            bt = bt[bt["index_name"].map(_match)]
        print("\n--- Backtest metrics ---")
        print(bt.to_string(index=False))
        if "index_name" in bt.columns and len(bt):
            print("\n--- Backtest by index (mean DirAcc / MAE) ---")
            # explode multi labels for summary
            rows = []
            for _, r in bt.iterrows():
                for lab in str(r.get("index_name", "")).split(","):
                    lab = lab.strip()
                    if lab:
                        rows.append({"index_name": lab, "directional_accuracy": r.get("directional_accuracy"), "mae": r.get("mae")})
            if rows:
                import numpy as np
                sm = pd.DataFrame(rows).groupby("index_name").agg(
                    n=("mae", "count"), mean_mae=("mae", "mean"), mean_diracc=("directional_accuracy", "mean")
                ).reset_index()
                print(sm.to_string(index=False))

    # Optional: vs last realized return for context
    if PRICES_FILE.exists():
        try:
            prices = pd.read_parquet(PRICES_FILE)
            prices["date"] = pd.to_datetime(prices["date"])
            print("\n--- Recent 20d realized vs forecast signal ---")
            for t in sorted(fc["ticker"].unique()):
                s = prices[prices["ticker"] == t].sort_values("date")["close"]
                if len(s) < 21:
                    continue
                realized = (s.iloc[-1] / s.iloc[-21] - 1) * 100
                exp = tail.loc[tail["ticker"] == t, "pct_change"]
                exp = float(exp.iloc[0]) if len(exp) else float("nan")
                print(f"  {t:6}  realized_20d={realized:+6.2f}%  forecast_H{max_h}={exp:+6.2f}%")
        except Exception:
            pass


if __name__ == "__main__":
    main()

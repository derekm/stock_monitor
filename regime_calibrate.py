#!/usr/bin/env python3
"""
regime_calibrate.py — two production-hardening jobs for regime-selected
Granite-TTM serving:

  1. CALIBRATION CHECK (default): verify the MC-dropout std band is honest.
     For each covered (ticker, regime), take the model's OOS test windows,
     run MC-dropout to get (mean, std), and check coverage of a z=1 band:
     coverage ≈ 68% is well-calibrated; far off means the band is lying.

  2. COVERAGE TRAIN (--train): batch-train regime checkpoints for tickers
     that have no pass6 coverage, so regime selection extends beyond
     AEP/NVR/FICO. One model per (ticker, regime) at the pass6-best config
     (steps=3000, cap=100, lr=None) — the most-common best config from
     pass7's matrix.

Why it exists: closes the "calibration check" and "extend coverage beyond
3 tickers" findings from the Granite-TTM review.

Usage:
    python regime_calibrate.py --tickers AEP,NVR          # calibration report
    python regime_calibrate.py --tickers MSFT,GOOG --train # train checkpoints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pass6
from pass6 import (tag_windows, temporal_split, train_regime_model,
                   REGIMES, MIN_TEST, GAP_DAYS, _channels_from_close)
from pass5 import P2_WIN, persistence_on_test
from regime_forecast import load_regime_map, clean_series_dated, windows_with_dates
from granite_backfill import gd
from regime_serving import CKPT_DIR, serve_regime_model, current_regime

CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
OUT_CAL = Path(__file__).parent / "regime_calibration.csv"


def _window_block(s, dates, lo, hi):
    out = []
    for wname, wp in P2_WIN.items():
        out += windows_with_dates(s, lo, hi, wp["stride"], wp["cap"], dates)
    return out


def calibration_report(tickers: list[str]) -> pd.DataFrame:
    """Coverage of the z=1 MC-dropout band on OOS test windows."""
    regime_s = load_regime_map()
    rows = []
    for tk in tickers:
        s, dates = clean_series_dated(tk)
        n = len(s)
        all_wins = _window_block(s, dates, 0, n)
        tagged = tag_windows(all_wins, regime_s, dates)
        by_regime = {r: [] for r in REGIMES}
        for w in tagged:
            if w[3] in by_regime:
                by_regime[w[3]].append(w)
        for reg in REGIMES:
            rw = by_regime.get(reg, [])
            if len(rw) < MIN_TEST + 10:
                continue
            train, test = temporal_split(rw, int(n * 0.7))
            if len(test) < MIN_TEST:
                continue
            # use the pass6 best config for this regime, or the default
            from regime_serving import best_config_for
            cfg = best_config_for(tk, reg) or {"steps": 3000, "cap": 100, "lr": None}
            tr_win = train
            if cfg.get("cap") and len(tr_win) > cfg["cap"]:
                idxs = np.linspace(0, len(tr_win) - 1, cfg["cap"]).astype(int)
                tr_win = [train[i] for i in idxs]
            tag = f"{tk}|{reg}|cal"
            r = pass6.train_regime_model(tr_win, test, cfg["steps"], tag,
                                         lr=cfg.get("lr"), n_channels=1)
            if r.get("skipped"):
                continue
            # MC-dropout z=1 band coverage on the first 10 OOS points
            import torch
            import copy
            from pass6 import BASE_MODEL, device
            from forecast_granite import forecast_ttm_mc_dropout
            m = copy.deepcopy(BASE_MODEL).to(device)
            m.eval()
            covered, n_pts = 0, 0
            for (c, t, *_ ) in test:
                mean, std = forecast_ttm_mc_dropout(m, "granite",
                                                    np.asarray(c, dtype=np.float32),
                                                    min(10, len(np.asarray(t))),
                                                    samples=8)
                if std is None:
                    continue
                a = np.asarray(t, dtype=np.float32)
                for h in range(min(10, len(a))):
                    if mean[h] - std[h] <= a[h] <= mean[h] + std[h]:
                        covered += 1
                    n_pts += 1
            rows.append({
                "ticker": tk, "regime": reg, "n_test_windows": len(test),
                "mc_band_cov_z1": round(covered / n_pts, 3) if n_pts else None,
                "expected_if_calibrated": 0.683,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default="AEP,NVR,FICO")
    ap.add_argument("--train", action="store_true", help="train checkpoints for uncovered tickers")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cap", type=int, default=100)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.train:
        reg = current_regime()
        if not reg:
            print("No regime label — cannot select checkpoints.")
            return
        trained, skipped = [], []
        for tk in tickers:
            path, cfg, reason = serve_regime_model(tk)
            if reason == "served":
                skipped.append((tk, "already served"))
                continue
            print(f"  training {tk} for regime {reg} (steps={args.steps} cap={args.cap})...")
            from regime_forecast import load_regime_map, clean_series_dated, windows_with_dates
            from pass6 import tag_windows, temporal_split, REGIMES
            regime_s = load_regime_map()
            s, dates = clean_series_dated(tk)
            n = len(s)
            all_wins = _window_block(s, dates, 0, n)
            tagged = tag_windows(all_wins, regime_s, dates)
            rw = [w for w in tagged if w[3] == reg]
            if len(rw) < MIN_TEST + 10:
                skipped.append((tk, f"only {len(rw)} {reg} windows"))
                continue
            train, test = temporal_split(rw, int(n * 0.7))
            tr_win = train
            if len(tr_win) > args.cap:
                idxs = np.linspace(0, len(tr_win) - 1, args.cap).astype(int)
                tr_win = [train[i] for i in idxs]
            tag = f"{tk}|{reg}|st={args.steps}|cap={args.cap}|lr=None"
            r = train_regime_model(tr_win, test, args.steps, tag,
                                   ckpt_dir=CKPT_DIR, n_channels=1)
            if r.get("skipped"):
                skipped.append((tk, "skipped"))
            else:
                trained.append((tk, r["dir_acc"], r["pers_dir"]))
                print(f"    {tk}: dir={r['dir_acc']}% (pers {r['pers_dir']}%) "
                      f"ckpt saved")
        print(f"\nTrained: {len(trained)}  Skipped: {len(skipped)}")
        for tk, d in trained:
            print(f"  {tk}: dir {d[1]}% vs pers {d[2]}%")
        for tk, why in skipped:
            print(f"  {tk}: {why}")
        return

    # calibration report — coverage of the MC-dropout z=1 band
    regime_s = load_regime_map()
    rows = []
    for tk in tickers:
        s, dates = clean_series_dated(tk)
        n = len(s)
        all_wins = _window_block(s, dates, 0, n)
        tagged = tag_windows(all_wins, regime_s, dates)
        by_regime = {r: [] for r in REGIMES}
        for w in tagged:
            if w[3] in by_regime:
                by_regime[w[3]].append(w)
        for reg in REGIMES:
            rw = by_regime.get(reg, [])
            if len(rw) < MIN_TEST + 10:
                continue
            train, test = temporal_split(rw, int(n * 0.7))
            if len(test) < MIN_TEST:
                continue
            cfg = pass6.__dict__.get("best_config_for") or {}
            # calibrate the point model's MC band on OOS test windows
            import torch
            from pass6 import BASE_MODEL, device
            from forecast_granite import _channels_from_series
            import copy
            m = copy.deepcopy(BASE_MODEL).to(device)
            m.eval()
            from forecast_granite import forecast_ttm_mc_dropout
            covered, n_pts = 0, 0
            for (c, t, *_ ) in test:
                x = np.asarray(c, dtype=np.float32)
                a = np.asarray(t, dtype=np.float32)
                mean, std = forecast_ttm_mc_dropout(m, "granite", x, min(10, len(a)),
                                                    samples=8)
                if std is None:
                    continue
                # z=1 band coverage on the first min(10, H) points
                for h in range(min(10, len(a))):
                    lo = mean[h] - std[h]
                    hi = mean[h] + std[h]
                    if lo <= a[h] <= hi:
                        covered += 1
                    n_pts += 1
            rows.append({
                "ticker": tk, "regime": reg, "n_test_windows": len(test),
                "mc_band_cov_z1": round(covered / n_pts, 3) if n_pts else None,
                "expected_if_calibrated": 0.683,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT_CAL, index=False)
        print(df.to_string(index=False))
        ok = df[df["mc_band_cov_z1"].notna()].assign(
            err=lambda d: (d["mc_band_cov_z1"] - 0.683).abs())
        if len(ok):
            print(f"\nmean |cov - 0.683| = {ok['err'].mean():.3f} "
                  f"({'well-calibrated' if ok['err'].mean() < 0.15 else 'POORLY calibrated'})")
        print(f"\nWrote {OUT_CAL}")
    else:
        print("No rows (insufficient windows?).")


if __name__ == "__main__":
    main()

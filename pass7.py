#!/usr/bin/env python3
"""
pass7.py — Experiment-design matrix for regime-selected Granite-TTM models.

pass6 found per-regime best (steps, cap) at ONE design point (70/30 split,
single lr, pure-regime training). pass7 tests whether those findings are
ROBUST by varying the experiment design itself — the "several different
experiment designs with different mixes" idea:

Arms (all evaluated with the same honesty rules as pass6 — shared global
boundary, embargo = HORIZON, per-regime persistence baseline, IBM base only):

  A. boundary     — split_frac in {0.55, 0.70, 0.85}: does the best config
                    survive a different train/test boundary?
  B. composition  — pure (train only on the regime's windows) vs all (train
                    on every window, evaluate per regime): does regime
                    specialization beat more training data?
  C. lr           — gd.LR (1e-4) vs 5e-5: is the finding lr-sensitive?
  D. freshness    — full in-regime history vs only the most recent N years
                    of in-regime windows: the "hold off trainings until the
                    trend switches back into that model's regime" test.

Output:
  /tmp/pass7_results.jsonl — append-only, resumable
  regime_model_matrix.csv  — every (arm, ticker, regime, config) result
  regime_model_matrix_summary.csv — per-arm mean/max OOS dir excess + config
    stability (how often each (steps, cap) wins across boundaries)

Usage:
    python pass7.py --tickers AEP --arms boundary --max-experiments 14
    python pass7.py --tickers AEP,NVR --arms composition,ln,freshness
    python pass7.py --resume --max-experiments 20
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

import pass4
import pass6
from pass6 import (tag_windows, temporal_split, train_regime_model,
                   REGIMES, MIN_TEST, GAP_DAYS)
from pass5 import P2_WIN, persistence_on_test
from regime_forecast import load_regime_map, clean_series_dated, windows_with_dates
from granite_backfill import gd

CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
OUT_JSONL = "/tmp/pass7_results.jsonl"
OUT_CSV = None
OUT_SUM = None
ARMS = ["boundary", "composition", "lr", "freshness"]


def _window_block(s, dates, lo, hi):
    out = []
    for wname, wp in P2_WIN.items():
        out += windows_with_dates(s, lo, hi, wp["stride"], wp["cap"], dates)
    return out


def run(tickers, arms, split_fracs, caps, steps_list, lrs, fresh_years,
        resume, max_experiments):
    global OUT_CSV, OUT_SUM
    data_dir = Path(__file__).resolve().parent
    OUT_CSV = data_dir / "regime_model_matrix.csv"
    OUT_SUM = data_dir / "regime_model_matrix_summary.csv"

    regime_s = load_regime_map()
    done = set()
    if resume and os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r.get("arm"), r.get("ticker"), r.get("regime"),
                          r.get("split_frac"), r.get("steps"), r.get("cap"),
                          r.get("lr"), r.get("fresh_years")))
            except Exception:
                pass
        print(f"resume: {len(done)} cells done", flush=True)

    results = []
    n_run = 0
    with open(OUT_JSONL, "a" if resume else "w") as f:
        for tk in tickers:
            s, dates = clean_series_dated(tk)
            n = len(s)
            all_wins = _window_block(s, dates, 0, n)
            tagged = tag_windows(all_wins, regime_s, dates)
            by_regime = {r: [] for r in REGIMES}
            for w in tagged:
                if w[3] in by_regime:
                    by_regime[w[3]].append(w)

            for reg in [r for r in REGIMES if r in (args.regimes or REGIMES)]:
                rw = by_regime.get(reg, [])
                if len(rw) < MIN_TEST + 10:
                    print(f"  {tk} {reg}: only {len(rw)} windows, skipping", flush=True)
                    continue
                # arm D: freshness — keep only the most recent in-regime windows
                rw_fresh = sorted(rw, key=lambda w: w[2])[-max(10, fresh_years * 252 // 5):]
                for sf in split_fracs:
                    boundary = int(n * sf)
                    train_pure, test = temporal_split(rw, boundary)
                    train_all, _ = temporal_split(tagged, boundary)
                    train_fresh, test_fresh = temporal_split(rw_fresh, boundary)
                    if len(test) < MIN_TEST:
                        continue
                    pers = persistence_on_test([(c, t) for c, t, *_ in test])
                    pers_dir = pers["dir_acc"] if pers else None
                    for steps in steps_list:
                        for cap in caps:
                            for lr in lrs:
                                comps = ["pure", "all"] if "composition" in arms else ["pure"]
                                if "freshness" in arms:
                                    comps.append("fresh")
                                for comp in comps:
                                    # which arms does this cell belong to?
                                    cell_arms = []
                                    if sf in (0.55, 0.70, 0.85):
                                        cell_arms.append("boundary")
                                    if comp == "all":
                                        cell_arms.append("composition")
                                    if lr is not None and lr < 1e-4:
                                        cell_arms.append("lr")
                                    if comp == "fresh":
                                        cell_arms.append("freshness")
                                    if not cell_arms:
                                        cell_arms = ["baseline"]
                                    tr_win = {"pure": train_pure, "all": train_all,
                                              "fresh": train_fresh}[comp]
                                    if cap and len(tr_win) > cap:
                                        idxs = np.linspace(0, len(tr_win) - 1, cap).astype(int)
                                        tr_win = [tr_win[i] for i in idxs]
                                    key = (sf, steps, cap, lr, comp)
                                    if resume and any(
                                            (a, tk, reg, sf, steps, cap, lr, None) in done
                                            for a in cell_arms):
                                        continue
                                    if max_experiments and n_run >= max_experiments:
                                        print(f"reached --max-experiments {max_experiments}", flush=True)
                                        _finish(results)
                                        return
                                    tag = f"{'+'.join(cell_arms)}|{tk}|{reg}|sf={sf}|st={steps}|cap={cap}|lr={lr}|{comp}"
                                    r = train_regime_model(tr_win, test, steps, tag, lr=lr)
                                    if r.get("skipped"):
                                        print(f"    {tag}: skipped", flush=True)
                                        continue
                                    r.update(arm="+".join(cell_arms), ticker=tk, regime=reg,
                                             split_frac=sf, steps=steps, cap=cap, lr=lr,
                                             composition=comp, pers_dir=pers_dir)
                                    results.append(r)
                                    f.write(json.dumps(r) + "\n")
                                    f.flush()
                                    n_run += 1
                                    print(f"    {tag}: dir={r['dir_acc']}% (pers {pers_dir}%) "
                                          f"[{r['secs']}s]", flush=True)
    _finish(results)


def _finish(results):
    if not results:
        print("No results.")
        return
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    df["excess"] = df["dir_acc"] - df["pers_dir"].fillna(50.0)
    # per-arm summary: mean/max excess, and config stability (best config freq)
    rows = []
    for arm, g in df.groupby("arm"):
        rows.append({
            "arm": arm, "n": len(g),
            "mean_excess": round(g["excess"].mean(), 2),
            "max_excess": round(g["excess"].max(), 2),
            "best_config_most_common": g.loc[g.groupby(["ticker", "regime"])["excess"].idxmax()]
                                          .groupby(["steps", "cap"])["excess"].agg("size")
                                          .idxmax() if len(g) else None,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUM, index=False)
    print("\n=== pass7 arm summary (mean/max OOS dir excess over persistence) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_SUM}")


def main():
    global args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default="AEP")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--split-fracs", nargs="+", type=float, default=[0.55, 0.70, 0.85])
    ap.add_argument("--caps", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--steps", nargs="+", type=int, default=[3000, 6000])
    ap.add_argument("--lrs", nargs="+", type=float, default=[None, 5e-5])
    ap.add_argument("--regimes", nargs="+", default=None)
    ap.add_argument("--fresh-years", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-experiments", type=int, default=None)
    args = ap.parse_args()
    if args.quick:
        args.tickers, args.arms = "AEP", ["boundary"]
        args.split_fracs, args.caps, args.steps, args.lrs = [0.7], [100], [300], [None]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run(tickers, args.arms, args.split_fracs, args.caps, args.steps, args.lrs,
        args.fresh_years, args.resume, args.max_experiments)


if __name__ == "__main__":
    main()

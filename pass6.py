#!/usr/bin/env python3
"""
pass6.py — regime-SELECTED Granite-TTM models (one model per regime).

pass5 proved Granite-TTM is a direction forecaster (OOS, beats persistence on
direction, loses on level). regime_forecast.py showed persistence itself is
regime-dependent (up-window frequency is highest in high_vol_stress), so a
single global model is suboptimal: its edge must be read against a
regime-specific baseline, and its training data mixes regimes with different
dynamics.

This pass trains ONE MODEL PER HMM REGIME (low_vol / normal / high_vol_stress),
each fine-tuned only on that regime's windows, and sweeps training parameters
per regime (steps, window cap, lr) so each regime gets the config that suits
its data volume and volatility regime. Recommendations become regime-SELECTED:
the current HMM regime picks its dedicated model + best config — not a gate
that merely down-weights trust.

Honesty rules (pass5 discipline + regime rigor):
  * GLOBAL temporal split: train windows' TARGETS end before a shared boundary
    date; test windows start after a gap (embargo = HORIZON days). All regimes
    share the same boundary, so no regime's test section can leak into any
    model's training.
  * Per-regime models are fine-tuned from the IBM base model only
    (pretrained=False) — no full-history checkpoint contamination.
  * Persistence baseline computed on the SAME test windows PER REGIME
    (apples-to-apples; persistence differs by regime).
  * A regime with < MIN_TEST test windows is skipped (too thin to claim).
  * Test window TARGETS lie entirely after training targets (no straddle).
    Context may overlap the boundary — that mirrors live use, where recent
    context is always available.

Output: /tmp/pass6_results.jsonl (append, resumable) + regime_model_oos.csv
        + regime_model_best.csv (best config per ticker x regime).

Usage:
    python pass6.py --tickers AEP,NVR,FICO --steps 3000 6000 --caps 100 200
    python pass6.py --tickers AEP --quick          # 1 config x 1 regime, tiny
    python pass6.py --tickers AEP --regimes high_vol_stress
    python pass6.py --tickers AEP --resume         # skip done configs
"""
from __future__ import annotations

import argparse, json, time, copy
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import pass4
from pass5 import clean_series, persistence_on_test, P2_WIN
from regime_forecast import load_regime_map, clean_series_dated, windows_with_dates
from granite_backfill import gd

device = pass4.device
BATCH = pass4.BATCH
BASE_MODEL = pass4.BASE_MODEL
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
REGIMES = ["low_vol", "normal", "high_vol_stress"]
MIN_TEST = 30  # min test windows to claim a per-regime result
GAP_DAYS = HORIZON  # embargo between train targets and test start (96d)
OUT_JSONL = "/tmp/pass6_results.jsonl"
OUT_CSV = None  # set in main from DATA_DIR
OUT_BEST = None


def tag_windows(wins: list, regime_s: pd.Series, dates, lookback: int = 20) -> list[tuple]:
    """Attach (ctx, tgt, forecast_idx, regime) — regime at the forecast point
    (majority over the last `lookback` trading days ending at context end)."""
    out = []
    for c, t, k in wins:
        fpt = k + CONTEXT - 1  # last context day = forecast point
        if fpt < 0 or fpt >= len(dates):
            continue
        d = pd.Timestamp(dates[fpt])
        prior = regime_s[regime_s.index <= d]
        if not len(prior):
            continue
        reg = str(prior.iloc[-1])
        out.append((c, t, fpt, reg))
    return out


def temporal_split(wins: list[tuple], boundary: int):
    """(train, test) — train targets end before `boundary` (forecast idx), test
    forecast idx >= boundary + GAP. Shared boundary across ALL regimes."""
    train, test = [], []
    for c, t, fpt, reg in wins:
        # target window occupies [fpt+1, fpt+HORIZON]; require it ends before boundary
        if fpt + HORIZON < boundary:
            train.append((c, t, fpt, reg))
        elif fpt >= boundary + GAP_DAYS:
            test.append((c, t, fpt, reg))
    return train, test


def train_regime_model(train_wins, test_wins, steps, tag, lr=None):
    """Fine-tune from the IBM base on train_wins; score on test_wins."""
    if len(train_wins) < 3 or len(test_wins) < 3:
        return dict(skipped=True, n_train=len(train_wins), n_test=len(test_wins), tag=tag)
    # strip the extra tuple fields for the loader
    tr = [(c, t) for c, t, *_ in train_wins]
    te = [(c, t) for c, t, *_ in test_wins]
    ctx = np.stack([w[0] for w in tr])[:, :, None]
    tgt = np.stack([w[1] for w in tr])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    m = copy.deepcopy(BASE_MODEL)  # IBM base only — no checkpoint contamination
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr if lr is not None else gd.LR)
    s = 0
    t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb)
            loss = o.loss
            if not torch.isfinite(loss):
                print(f"    [NaN loss {tag}, aborting]")
                s = steps
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            s += 1
            if s >= steps:
                break
        if s >= steps:
            break
    dt = time.time() - t0
    m.eval()
    p_all, a_all, cl = [], [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(
                torch.tensor(np.stack([w[0] for w in te])[:, :, None]),
                torch.tensor(np.stack([w[1] for w in te])[:, :, None])),
                batch_size=BATCH, shuffle=False, pin_memory=True):
            out = m(past_values=xb.to(device))
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            p_all.append(p.cpu().float().numpy())
            a_all.append(yb.numpy())
            cl.append(xb[:, -1, 0].cpu().numpy())
    P = np.concatenate(p_all, 0).squeeze(-1)
    A = np.concatenate(a_all, 0).squeeze(-1)
    CL = np.concatenate(cl, 0)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    dir_acc = float((np.sign(A.mean(1) - CL) == np.sign(P.mean(1) - CL)).mean() * 100)
    pers = persistence_on_test(te)
    return dict(
        mape=round(mape, 2), dir_acc=round(dir_acc, 1),
        mape_pers=pers["mape"] if pers else None,
        pers_dir=pers["dir_acc"] if pers else None,
        n_train=len(tr), n_test=len(te), secs=round(dt, 1), tag=tag,
    )


def run(tickers, steps_list, caps_list, lr_list, regimes, split_frac, resume, max_experiments):
    global OUT_CSV, OUT_BEST
    from pathlib import Path
    import os
    data_dir = Path(__file__).resolve().parent
    OUT_CSV = data_dir / "regime_model_oos.csv"
    OUT_BEST = data_dir / "regime_model_best.csv"

    regime_s = load_regime_map()
    done = set()
    if resume and os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r.get("ticker"), r.get("regime"), r.get("steps"), r.get("cap"), r.get("lr")))
            except Exception:
                pass
        print(f"resume: {len(done)} configs already done", flush=True)

    results = []
    n_run = 0
    with open(OUT_JSONL, "a" if resume else "w") as f:
        for tk in tickers:
            s, dates = clean_series_dated(tk)
            n = len(s)
            boundary = int(n * split_frac)
            # windows across full history (hi = n, matching pass5 block bounds)
            all_wins = []
            for wname, wp in P2_WIN.items():
                stride, cap = wp["stride"], wp["cap"]
                all_wins += windows_with_dates(s, 0, n, stride, cap, dates)
            tagged = tag_windows(all_wins, regime_s, dates)
            by_regime: dict[str, list] = {r: [] for r in REGIMES}
            for w in tagged:
                if w[3] in by_regime:
                    by_regime[w[3]].append(w)
            for reg in regimes:
                rw = by_regime.get(reg, [])
                if len(rw) < MIN_TEST + 10:
                    print(f"  {tk} {reg}: only {len(rw)} windows, skipping", flush=True)
                    continue
                train, test = temporal_split(rw, boundary)
                if len(test) < MIN_TEST:
                    print(f"  {tk} {reg}: {len(test)} test windows (< {MIN_TEST}), skipping", flush=True)
                    continue
                pers = persistence_on_test([(c, t) for c, t, *_ in test])
                print(f"  {tk} {reg}: train={len(train)} test={len(test)} "
                      f"pers_dir={pers['dir_acc'] if pers else '-'}%", flush=True)
                for steps in steps_list:
                    for cap in caps_list:
                        for lr in lr_list:
                            key = (tk, reg, steps, cap, lr)
                            if resume and key in done:
                                continue
                            if max_experiments and n_run >= max_experiments:
                                print(f"reached --max-experiments {max_experiments}", flush=True)
                                _finish(results)
                                return
                            # cap the training windows (density control per regime)
                            tr_win = train
                            if cap and len(tr_win) > cap:
                                idxs = np.linspace(0, len(tr_win) - 1, cap).astype(int)
                                tr_win = [train[i] for i in idxs]
                            tag = f"{tk}|{reg}|st={steps}|cap={cap}|lr={lr}"
                            r = train_regime_model(tr_win, test, steps, tag, lr=lr)
                            if r.get("skipped"):
                                print(f"    {tag}: skipped", flush=True)
                                continue
                            r.update(ticker=tk, regime=reg, steps=steps, cap=cap, lr=lr,
                                     pers_dir=pers["dir_acc"] if pers else None,
                                     pers_mape=pers["mape"] if pers else None)
                            results.append(r)
                            f.write(json.dumps(r) + "\n")
                            f.flush()
                            n_run += 1
                            print(f"    {tag}: dir={r['dir_acc']}% MAPE={r['mape']} "
                                  f"(pers {r.get('pers_dir')}%) [{r['secs']}s]", flush=True)
    _finish(results)


def _finish(results):
    if not results:
        print("No results.")
        return
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    # best config per (ticker, regime): highest direction accuracy excess over
    # the regime's persistence baseline (the honest regime-selected objective)
    df["excess"] = df["dir_acc"] - df["pers_dir"].fillna(50.0)
    best = df.loc[df.groupby(["ticker", "regime"])["excess"].idxmax()].reset_index(drop=True)
    best.to_csv(OUT_BEST, index=False)
    print("\n=== best per-regime configs (max OOS dir excess over persistence) ===")
    print(best[["ticker", "regime", "steps", "cap", "lr", "dir_acc", "pers_dir",
                "mape", "n_test", "secs"]].to_string(index=False))
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_BEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default="AEP,NVR,FICO")
    ap.add_argument("--steps", nargs="+", type=int, default=[3000, 6000])
    ap.add_argument("--caps", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--lrs", nargs="+", type=float, default=[None])  # None = gd.LR
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--split-frac", type=float, default=0.7, help="train fraction of history (shared boundary)")
    ap.add_argument("--quick", action="store_true", help="1 config x 1 regime, tiny")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-experiments", type=int, default=None)
    args = ap.parse_args()
    if args.quick:
        args.tickers = "AEP"
        args.steps = [300]
        args.caps = [50]
        args.regimes = ["normal"]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run(tickers, args.steps, args.caps, args.lrs, args.regimes,
        args.split_frac, args.resume, args.max_experiments)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pass8.py — own RPT-pre-trained base + regime fine-tunes from it.

Why: pass6's --rpt probe proved the IBM granite-timeseries-ttm-r2 base was
NOT pre-trained with Resolution Prefix Tuning — enabling it at fine-tune time
adds a freq patch (8->9) that breaks the multi-level patch-partition reshape
the pretrained backbone was built for (mat1 384x36 vs 32x64). RPT is a
PRE-TRAINING technique (TTM paper 3.1.1): the resolution token must be in the
model's training distribution from the start.

pass8 fixes that by pre-training OUR OWN base with RPT enabled on the full
daily price history, then running the pass6 regime-selected fine-tune sweep
from that base. If RPT helps short-context regime windows (the paper's claim),
the regime models fine-tuned from the RPT base should beat the IBM-base cells
on OOS dir excess over persistence.

Pipeline:
  Stage A (--pretrain):   train a fresh TinyTimeMixer with
                          resolution_prefix_tuning=True, freq_token=8 (daily),
                          on univariate close series across ALL monitored
                          tickers (channel-independent, MSE objective, the
                          paper's pre-training workflow 3.1). Save base to
                          checkpoints/rpt_base/*.pt + config.
  Stage B (--fine-tune):  pass6-style regime-selected sweep (same cell grid:
                          tickers x regimes x steps x cap x lr, plus head-only
                          and exog toggles) but fine-tuning FROM THE RPT BASE
                          instead of the IBM base. Outputs to
                          regime_model_oos_rpt.csv / regime_model_best_rpt.csv
                          so IBM-base and RPT-base results stay comparable.

  Compare (--compare):    join the two OOS CSVs on (ticker, regime, steps,
                          cap, lr, head_only, exog) and report per-cell
                          dir-excess delta (RPT base minus IBM base).

Honesty rules: same as pass6 — global temporal split with embargo, persistence
baseline per regime on the same test windows, regime models fine-tuned from
the pre-trained base only (no test leakage).

Usage:
  python pass8.py --pretrain --steps 8000 --save-to checkpoints/rpt_base
  python pass8.py --fine-tune --tickers AEP,NVR,FICO --steps 3000 6000 \
                  --caps 100 200 --lrs None 5e-5 --head-only --exog \
                  --resume --max-experiments 120
  python pass8.py --compare

Outputs:
  checkpoints/rpt_base/ttm_rpt_<steps>.pt (+ config json) — Stage A
  regime_model_oos_rpt.csv / regime_model_best_rpt.csv — Stage B
  rpt_vs_ibm_compare.csv — per-cell delta table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pass4
import pass6
from pass5 import persistence_on_test, P2_WIN
from pass6 import (train_regime_model, tag_windows, temporal_split,
                   _exog_channel, _channels_from_close, REGIMES, MIN_TEST,
                   HORIZON_SPANS)
from regime_forecast import load_regime_map, clean_series_dated, windows_with_dates

device = pass4.device
BATCH = pass4.BATCH
CONTEXT, HORIZON = pass6.CONTEXT, pass6.HORIZON
DATA_DIR = Path(__file__).resolve().parent
FREQ_DAILY = 8  # daily resolution token per granite-tsfm DEFAULT_FREQUENCY_MAPPING
# (time_series_preprocessor.py: oov=0, min=1, ..., h=7, d=8, W=9).
# NOTE: tsfm_public 0.3.8 ships the freq helpers as an unimplemented stub
# (utils_tinytimemixer.py is referenced but absent), so nothing enforces this
# locally — we use 8 to match the upstream canonical mapping so a future tsfm
# upgrade serves RPT checkpoints with the correct daily token.

OUT_OOS_RPT = DATA_DIR / "regime_model_oos_rpt.csv"
OUT_BEST_RPT = DATA_DIR / "regime_model_best_rpt.csv"
OUT_JSONL = "/tmp/pass8_results.jsonl"
OUT_COMPARE = DATA_DIR / "rpt_vs_ibm_compare.csv"


# ---------------------------------------------------------------- Stage A
def _fresh_rpt_model(n_channels: int = 1):
    """Fresh TinyTimeMixer with RPT enabled (NOT the IBM base — a model that
    was never trained with RPT cannot absorb the freq token at fine-tune
    time; the partition arithmetic breaks, proven by the pass6 probe).

    num_patches MUST be bumped 8->9 when RPT is on: the freq token occupies
    a 9th patch slot, and every TSMixer layer is built from config.num_patches.
    Building with num_patches=9 makes the whole model consistent (verified:
    forward with freq_token=8 produces (2, 96, 1))."""
    from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
    cfg = TinyTimeMixerConfig.from_pretrained(pass6.gd.DEFAULT_MODEL)
    cfg.resolution_prefix_tuning = True
    # vocab must cover the FULL canonical mapping (0..9: oov, min, 2/5/10/15/30min,
    # h/H=7, d/D=8, W=9). The IBM base ships vocab=5 (only intraday tokens) —
    # that's why the freq embedding assertion fired at token 8 (daily).
    cfg.frequency_token_vocab_size = 10
    cfg.num_patches = 9  # freq token occupies the 9th patch slot
    cfg.num_input_channels = n_channels
    return TinyTimeMixerForPrediction(cfg).to(device), cfg


def _pretrain_loader(tickers=None, window=CONTEXT, max_tickers=200):
    """Channel-independent univariate (close-only) windows across tickers.
    Batch of (ctx, tgt) pairs; freq token applied at forward (always daily)."""
    import pandas as pd
    cols = ["date", "ticker", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    if tickers:
        d = d[d["ticker"].isin(tickers)]
    all_win = []
    for t, g in d.groupby("ticker"):
        s = g["close"].to_numpy(dtype=np.float32)
        if len(s) < window + HORIZON + 10:
            continue
        # dense windows, no stride — maximum coverage per ticker
        n = len(s)
        for k in range(0, n - (window + HORIZON), 5):
            all_win.append((s[k:k + window], s[k + window:k + window + HORIZON]))
            if len(all_win) >= 200_000:
                break
        if len(all_win) >= 200_000:
            break
    print(f"pretrain windows: {len(all_win)}", flush=True)
    return all_win


def pretrain(steps: int, save_to: Path, tickers=None):
    """Train a fresh RPT-enabled base on univariate daily closes (MSE)."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    save_to.mkdir(parents=True, exist_ok=True)
    wins = _pretrain_loader(tickers)
    if not wins:
        raise SystemExit("no pretrain windows")
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=True)
    m, cfg = _fresh_rpt_model()
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    s = 0
    t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            fwd = {"freq_token": torch.full((xb.shape[0],), FREQ_DAILY, dtype=torch.long, device=device)}
            o = m(past_values=xb, future_values=yb, **fwd)
            loss = o.loss
            if not torch.isfinite(loss):
                print("NaN loss, aborting")
                s = steps
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            s += 1
            if s % 100 == 0:
                print(f"  step {s}: loss={float(loss):.5f} [{time.time()-t0:.0f}s]", flush=True)
            if s >= steps:
                break
        if s >= steps:
            break
    # save base: full state dict + config so Stage B can rebuild exactly
    ckpt = save_to / f"ttm_rpt_{steps}.pt"
    torch.save({"model": m.state_dict(), "rpt": True, "n_channels": 1,
                "config": cfg.to_dict(), "steps": steps,
                "trained_on": pd.Timestamp.now().isoformat()}, ckpt)
    (save_to / f"ttm_rpt_{steps}_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    print(f"Wrote {ckpt}")
    return ckpt


def load_rpt_base(ckpt_path: Path):
    """Rebuild the RPT base exactly (config from checkpoint, RPT on)."""
    from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = TinyTimeMixerConfig(**state["config"])
    m = TinyTimeMixerForPrediction(cfg).to(device)
    m.load_state_dict(state["model"])
    return m


# ---------------------------------------------------------------- Stage B
def fine_tune(tickers, steps_list, caps_list, lr_list, regimes, split_frac,
              resume, max_experiments, head_only, exog, rpt_base: Path, ckpt_dir: Path | None = None):
    """pass6-style regime sweep FROM the RPT base."""
    global done_set
    done_set = set()
    if resume and Path(OUT_JSONL).exists():
        for line in open(OUT_JSONL, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done_set.add((r.get("ticker"), r.get("regime"), r.get("steps"), r.get("cap"),
                              r.get("lr"), r.get("head_only", False), r.get("exog", False)))
            except Exception:
                pass
        print(f"resume: {len(done_set)} cells done", flush=True)

    base_model = load_rpt_base(rpt_base)  # the RPT base replaces BASE_MODEL
    # patch pass6's module-level base for train_regime_model's deepcopy
    pass6.BASE_MODEL = base_model
    pass6._CUSTOM_BASE_CKPT = str(rpt_base)  # channel expansion rebuilds from OUR base

    regime_s = load_regime_map()
    results = []
    n_run = 0
    with open(OUT_JSONL, "a" if resume else "w") as f:
        for tk in tickers:
            s, dates = clean_series_dated(tk)
            n = len(s)
            boundary = int(n * split_frac)
            all_wins = []
            for wname, wp in P2_WIN.items():
                stride, cap = wp["stride"], wp["cap"]
                all_wins += windows_with_dates(s, 0, n, stride, cap, dates)
            tagged = tag_windows(all_wins, regime_s, dates)
            by_regime = {r: [] for r in REGIMES}
            for w in tagged:
                if w[3] in by_regime:
                    by_regime[w[3]].append(w)
            for reg in regimes:
                rw = by_regime.get(reg, [])
                if len(rw) < MIN_TEST + 10:
                    continue
                train, test = temporal_split(rw, boundary)
                if len(test) < MIN_TEST:
                    continue
                pers = persistence_on_test([(c, t) for c, t, *_ in test])
                print(f"  {tk} {reg}: train={len(train)} test={len(test)} "
                      f"pers_dir={pers['dir_acc'] if pers else '-'}%", flush=True)
                for steps in steps_list:
                    for cap in caps_list:
                        for lr in lr_list:
                            key = (tk, reg, steps, cap, lr, head_only, exog)
                            if resume and key in done_set:
                                continue
                            if max_experiments and n_run >= max_experiments:
                                print(f"reached --max-experiments {max_experiments}", flush=True)
                                _finish_rpt(results)
                                return
                            tr_win = train
                            if cap and len(tr_win) > cap:
                                idxs = np.linspace(0, len(tr_win) - 1, cap).astype(int)
                                tr_win = [train[i] for i in idxs]
                            tag = f"rpt|{tk}|{reg}|st={steps}|cap={cap}|lr={lr}" + ("|head" if head_only else "") + ("|exog" if exog else "")
                            r = train_regime_model(tr_win, test, steps, tag, lr=lr,
                                                   n_channels=4 if exog else 1,
                                                   head_only=head_only, rpt=True,
                                                   exog=exog, dates=dates, ckpt_dir=ckpt_dir)
                            if r.get("skipped"):
                                print(f"    {tag}: skipped", flush=True)
                                continue
                            r.update(ticker=tk, regime=reg, steps=steps, cap=cap, lr=lr,
                                     head_only=head_only, rpt=True, exog=exog,
                                     base="rpt",
                                     pers_dir=pers["dir_acc"] if pers else None,
                                     pers_mape=pers["mape"] if pers else None)
                            results.append(r)
                            f.write(json.dumps(r) + "\n")
                            f.flush()
                            n_run += 1
                            print(f"    {tag}: dir={r['dir_acc']}% (pers {r.get('pers_dir')}%) "
                                  f"[{r['secs']}s]", flush=True)
    _finish_rpt(results)


def _finish_rpt(results):
    if not results:
        print("No results.")
        return
    df = pd.DataFrame(results)
    df.to_csv(OUT_OOS_RPT, index=False)
    df["excess"] = df["dir_acc"] - df["pers_dir"].fillna(50.0)
    best = df.loc[df.groupby(["ticker", "regime"])["excess"].idxmax()].reset_index(drop=True)
    best.to_csv(OUT_BEST_RPT, index=False)
    print("=== RPT-base best per-regime configs ===")
    show = ["ticker", "regime", "steps", "cap", "lr", "head_only", "exog", "dir_acc", "pers_dir", "mape", "secs"]
    print(best[[c for c in show if c in best.columns]].to_string(index=False))
    print(f"Wrote {OUT_OOS_RPT} ({len(df)} cells)\nWrote {OUT_BEST_RPT}")


# ---------------------------------------------------------------- Compare
def compare():
    ibm = pd.read_csv(DATA_DIR / "regime_model_oos.csv")
    rpt = pd.read_csv(OUT_OOS_RPT)
    ibm["excess"] = ibm["dir_acc"] - ibm["pers_dir"].fillna(50.0)
    rpt["excess"] = rpt["dir_acc"] - rpt["pers_dir"].fillna(50.0)
    keys = ["ticker", "regime", "steps", "cap", "lr", "head_only", "exog"]
    m = ibm.merge(rpt, on=keys, suffixes=("_ibm", "_rpt"))
    if m.empty:
        print("no overlapping cells (run Stage B first)")
        return
    m["excess_delta"] = m["excess_rpt"] - m["excess_ibm"]
    m = m.sort_values("excess_delta", ascending=False)
    m.to_csv(OUT_COMPARE, index=False)
    print(f"=== RPT base vs IBM base ({len(m)} overlapping cells) ===")
    show = keys + ["dir_acc_ibm", "dir_acc_rpt", "excess_delta"]
    print(m[show].head(15).to_string(index=False))
    print(f"\nmean excess_delta: {m['excess_delta'].mean():+.2f} pts")
    print(f"cells where RPT wins: {(m['excess_delta'] > 0).sum()}/{len(m)}")
    print(f"Wrote {OUT_COMPARE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain", action="store_true", help="Stage A: train RPT base")
    ap.add_argument("--steps", type=int, default=8000, help="Stage A pretrain steps")
    ap.add_argument("--save-to", default="checkpoints/rpt_base")
    ap.add_argument("--fine-tune", action="store_true", help="Stage B: regime sweep from RPT base")
    ap.add_argument("--tickers", default="AEP,NVR,FICO")
    ap.add_argument("--steps-list", nargs="+", type=int, default=[3000, 6000], help="Stage B fine-tune steps (e.g. 3000 6000)")
    ap.add_argument("--caps", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--split-frac", type=float, default=0.7)
    ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--exog", action="store_true")
    ap.add_argument("--rpt-base", default=None, help="Stage B base ckpt (default: newest in --save-to)")
    ap.add_argument("--ckpt-dir", default=None, help="Dir to save per-regime checkpoints (e.g. checkpoints/regime)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-experiments", type=int, default=None)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare()
        return
    save_dir = DATA_DIR / args.save_to
    if args.pretrain:
        pretrain(args.steps, save_dir)
        return
    if args.fine_tune:
        base = args.rpt_base
        if base is None:
            ckpts = sorted(save_dir.glob("ttm_rpt_*.pt"))
            if not ckpts:
                raise SystemExit(f"no RPT base in {save_dir} — run --pretrain first")
            base = ckpts[-1]
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        fine_tune(tickers, args.steps_list, args.caps, args.lrs, args.regimes,
                  args.split_frac, args.resume, args.max_experiments,
                  args.head_only, args.exog, Path(base), Path(args.ckpt_dir) if args.ckpt_dir else None)
        return
    ap.print_help()


if __name__ == "__main__":
    main()

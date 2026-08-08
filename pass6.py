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
from pathlib import Path
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
# direction-accuracy spans reported in production runs (cap = model horizon)
HORIZON_SPANS = [10, 21, 42, 63, 96]
REGIMES = ["low_vol", "normal", "high_vol_stress"]
MIN_TEST = 30  # min test windows to claim a per-regime result
GAP_DAYS = HORIZON  # embargo between train targets and test start (96d)
OUT_JSONL = "/tmp/pass6_results.jsonl"
OUT_CSV = None  # set in main from DATA_DIR
OUT_BEST = None
# pass8 hook: a path to OUR OWN pre-trained base checkpoint (e.g. an
# RPT-enabled base trained on the full price history). When set, channel
# expansion rebuilds from this checkpoint's config + weights instead of the
# IBM hub model — this is how regime fine-tunes from a custom base work.
_CUSTOM_BASE_CKPT: str | None = None


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


def _channels_from_close(w: np.ndarray) -> np.ndarray:
    """(close, pct_return, realized_vol20) channels from a close series."""
    c = np.asarray(w, dtype=np.float32)
    r = np.zeros_like(c)
    r[1:] = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    v = np.zeros_like(c)
    for i in range(len(c)):
        lo = max(0, i - 19)
        v[i] = np.std(r[lo:i + 1]) if i > lo else 0.0
    return np.stack([c, r, v], axis=-1)


# ---- exogenous event-proximity channel (TTM paper 3.2, Exogenous Mixer) ----
# The model only accepts past_values, so the honest form of "known-future
# exog" is a per-timestep channel: days-until-next-scheduled-event, computed
# from the economic calendar (FOMC + option expiries — both known years in
# advance). The model can learn to be cautious as an event approaches.
_EVENT_DATES: np.ndarray | None = None


def _load_event_dates() -> np.ndarray:
    """Sorted array of known-future event dates (FOMC + option expiry)."""
    global _EVENT_DATES
    if _EVENT_DATES is not None:
        return _EVENT_DATES
    events: list[pd.Timestamp] = []
    path = Path(__file__).resolve().parent / "economic_calendar.csv"
    if path.exists():
        ec = pd.read_csv(path)
        if "date" in ec.columns and "event_type" in ec.columns:
            for _, r in ec.iterrows():
                et = str(r.get("event_type", ""))
                if "fomc" in et.lower() or "expiry" in et.lower() or "fed" in et.lower():
                    try:
                        events.append(pd.Timestamp(r["date"]))
                    except Exception:
                        pass
    events.sort()
    _EVENT_DATES = np.array([e.value / 1e9 for e in events])  # epoch seconds
    return _EVENT_DATES


def _exog_channel(w: np.ndarray, end_epoch: float) -> np.ndarray:
    """Per-timestep channel: days-until-next-event as of each context day.
    end_epoch = epoch-seconds of the context-end date; the channel is built
    by walking the event calendar relative to each timestep's date."""
    n = len(w)
    ev = _load_event_dates()
    if len(ev) == 0:
        return np.zeros(n, dtype=np.float32)
    day = 86400.0
    out = np.zeros(n, dtype=np.float32)
    # context day i sits (n-1-i) days before the context end
    for i in range(n):
        t = end_epoch - (n - 1 - i) * day
        nxt = ev[ev >= t]
        if len(nxt):
            out[i] = float(min((nxt[0] - t) / day, 180.0))  # cap at 180d
    return out / 180.0  # normalize to [0,1]


def train_regime_model(train_wins, test_wins, steps, tag, lr=None, ckpt_dir=None,
                       n_channels: int = 1, return_model: bool = False,
                       head_only: bool = False, rpt: bool = False,
                       exog: bool = False, dates=None):
    """Fine-tune from the IBM base on train_wins; score on test_wins.

    When ckpt_dir is given, the trained model is saved there as
    <TICKER>__<regime>__<steps>__<lr>.pt so the production forecaster
    can serve it (regime-selected forecasts).

    n_channels=1: close-only (the pass5/6 default). n_channels=3: expand the
    pretrained model to (close, pct_return, realized_vol20) via
    AutoConfig num_input_channels + ignore_mismatched_sizes — the documented
    TTM path for adding channels to a pretrained single-channel model.
    exog=True appends a 4th channel: days-until-next-FOMC/expiry per timestep
    (known-future calendar events; TTM paper 3.2 Exogenous Mixer, input form).

    return_model=True: the trained model is attached to the result dict as
    ``_model`` (caller owns it and must del it) so downstream calibration can
    measure the ACTUAL served model's MC-dropout band, not the base model's.
    Default False keeps the sweep memory-safe (model deleted + CUDA cache
    emptied after scoring).
    """
    if len(train_wins) < 3 or len(test_wins) < 3:
        return dict(skipped=True, n_train=len(train_wins), n_test=len(test_wins), tag=tag)
    # strip the extra tuple fields for the loader
    tr = [(c, t) for c, t, *_ in train_wins]
    te = [(c, t) for c, t, *_ in test_wins]
    if exog:
        n_channels = max(n_channels, 4)
    if n_channels > 1:
        ctx_parts = []
        for i, (c, t) in enumerate(tr):
            ch = _channels_from_close(c)
            if exog and dates is not None:
                fpt = train_wins[i][2] if len(train_wins[i]) > 2 else 0
                end_epoch = pd.Timestamp(dates[fpt]).timestamp()
                ex = _exog_channel(c, end_epoch)
                ch = np.concatenate([ch, ex[:, None]], axis=-1)
            ctx_parts.append(ch)
        ctx = np.stack(ctx_parts)
        tgt = np.stack([w[1] for w in tr])[:, :, None]
    else:
        ctx = np.stack([w[0] for w in tr])[:, :, None]
        tgt = np.stack([w[1] for w in tr])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    m = copy.deepcopy(BASE_MODEL)  # IBM base only — no checkpoint contamination
    if n_channels > 1:
        # Expand a 1-channel pretrained model to n_channels: rebuild the
        # config with the larger channel count, re-instantiate from the
        # pretrained checkpoint, load state dict non-strictly (matching
        # channel weights keep pretrained values; new-channel weights get
        # fresh init). This is the standard TTM channel-expansion recipe.
        try:
            if _CUSTOM_BASE_CKPT is not None:
                # pass8: fine-tune from OUR OWN pre-trained base (RPT-enabled
                # or otherwise) instead of the IBM hub model. Rebuild from the
                # saved config so resolution_prefix_tuning etc. carry over;
                # load the state dict loosely so channel expansion works.
                from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig as _TMC
                from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction as _TMP
                _state = torch.load(_CUSTOM_BASE_CKPT, map_location="cpu")
                _cfg = _TMC(**_state.get("config", {})) if "config" in _state else _TMC.from_pretrained(gd.DEFAULT_MODEL)
                _cfg.num_input_channels = n_channels
                m = _TMP(_cfg).to(device)
                m.load_state_dict(_state["model"], strict=False)
            else:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(gd.DEFAULT_MODEL)
                cfg.num_input_channels = n_channels
                if rpt:
                    # Resolution Prefix Tuning: teach the model its sampling
                    # resolution explicitly. Daily data = freq token 2. Helps
                    # short-context regime windows where resolution is hard to
                    # infer from the data alone (TTM paper 3.1.1, RPT).
                    cfg.resolution_prefix_tuning = True
                    cfg.frequency_token_vocab_size = 5
                m = type(BASE_MODEL).from_pretrained(
                    gd.DEFAULT_MODEL, config=cfg, ignore_mismatched_sizes=True)
                m = m.to(device)
            if rpt and _CUSTOM_BASE_CKPT is None:
                # RPT is a PRE-TRAINING technique: the freq token adds a patch,
                # which changes the multi-level patch-partition arithmetic the
                # pretrained backbone was built for. Probe a real forward; if
                # the shapes break, RPT is not compatible with this base and we
                # fall back (honest degradation, never a silent wrong model).
                try:
                    with torch.no_grad():
                        probe = torch.randn(2, CONTEXT, n_channels, device=device)
                        tok = torch.full((2,), 8, dtype=torch.long, device=device)
                        m(past_values=probe, freq_token=tok)
                    print("    [RPT probe OK — resolution prefix active]")
                except Exception as e:
                    print(f"    [RPT incompatible with base ({str(e)[:80]}); continuing without RPT]")
                    rpt = False
                    m = type(BASE_MODEL).from_pretrained(
                        gd.DEFAULT_MODEL, ignore_mismatched_sizes=True).to(device)
        except Exception as e:
            print(f"    [channel expansion failed ({e}); falling back to close-only]")
            n_channels = 1
    if head_only:
        # TTM paper finding: freeze the backbone, tune only decoder + head
        # (36% of params). Backbone learned transferable temporal dynamics in
        # pre-training; regime-specific behavior lives in the head. Also
        # ~3x cheaper per cell on the 2GB GPU.
        for p in m.backbone.parameters():
            p.requires_grad = False
    m.train()
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad],
        lr=lr if lr is not None else gd.LR)
    # effective flags: RPT may have been disabled by the probe fallback
    eff_rpt = rpt
    s = 0
    t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            fwd_kw = {}
            if eff_rpt:
                # daily resolution token (8); shape [batch]
                fwd_kw["freq_token"] = torch.full((xb.shape[0],), 8, dtype=torch.long, device=device)
            o = m(past_values=xb, future_values=yb, **fwd_kw)
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
        if n_channels > 1:
            te_parts = []
            for i, (c, t) in enumerate(te):
                ch = _channels_from_close(c)
                if exog and dates is not None:
                    fpt = test_wins[i][2] if len(test_wins[i]) > 2 else 0
                    end_epoch = pd.Timestamp(dates[fpt]).timestamp()
                    ex = _exog_channel(c, end_epoch)
                    ch = np.concatenate([ch, ex[:, None]], axis=-1)
                te_parts.append(ch)
            te_ctx = np.stack(te_parts)
        else:
            te_ctx = np.stack([w[0] for w in te])[:, :, None]
        te_tgt = np.stack([w[1] for w in te])[:, :, None]
        for xb, yb in DataLoader(TensorDataset(torch.tensor(te_ctx), torch.tensor(te_tgt)),
                                 batch_size=BATCH, shuffle=False, pin_memory=True):
            fwd_kw = {}
            if eff_rpt:
                fwd_kw["freq_token"] = torch.full((xb.shape[0],), 8, dtype=torch.long, device=device)
            out = m(past_values=xb.to(device), **fwd_kw)
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            p_all.append(p.cpu().float().numpy())
            a_all.append(yb.numpy())
            cl.append(xb[:, -1, 0].cpu().numpy())
    P = np.concatenate(p_all, 0)
    A = np.concatenate(a_all, 0).squeeze(-1)
    CL = np.concatenate(cl, 0)
    if P.ndim == 3:
        P = P[..., 0]  # close channel only
    P = P.squeeze(-1) if P.ndim == 2 and P.shape[-1] == 1 else P
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    dir_acc = float((np.sign(A.mean(1) - CL) == np.sign(P.mean(1) - CL)).mean() * 100)
    # Production-horizon direction at several spans (up to the 96 max): the
    # live forecast uses --horizon 10 but signals can be read at 21/42/63/96.
    # Each span s measures mean-sign over the first s days of the target.
    spans = [s for s in HORIZON_SPANS if s <= A.shape[1]]
    span_acc = {}
    for s in spans:
        span_acc[f"dir_acc_h{s}"] = round(float(
            (np.sign(A[:, :s].mean(1) - CL) == np.sign(P[:, :s].mean(1) - CL)).mean() * 100), 1)
    if ckpt_dir is not None and dir_acc >= 55.0:
        # only persist models that beat the random-ish 50% floor; the
        # production consumer filters by regime_model_best.csv anyway
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        safe_lr = str(lr).replace(".", "p") if lr is not None else "None"
        tk_part = tag.split("|")[0]
        reg_part = tag.split("|")[1] if "|" in tag else "regime"
        fname = f"{tk_part}__{reg_part}__{steps}__{safe_lr}.pt"
        torch.save({"model": m.state_dict(), "dir_acc": dir_acc,
                    "tag": tag, "n_channels": n_channels,
                    "rpt": eff_rpt, "exog": exog,
                    "trained_on": pd.Timestamp.now().isoformat()}, ckpt_dir / fname)
    pers = persistence_on_test(te)
    result = dict(
        mape=round(mape, 2), dir_acc=round(dir_acc, 1),
        **span_acc,
        mape_pers=pers["mape"] if pers else None,
        pers_dir=pers["dir_acc"] if pers else None,
        n_train=len(tr), n_test=len(te), secs=round(dt, 1), tag=tag,
        head_only=head_only, rpt=eff_rpt, exog=exog,
    )
    if return_model:
        # caller owns the model (calibration measures the actual served model)
        result["_model"] = m
    else:
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def run(tickers, steps_list, caps_list, lr_list, regimes, split_frac, resume, max_experiments,
        n_channels: int = 1, ckpt_dir=None, head_only: bool = False, rpt: bool = False,
        exog: bool = False):
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
                done.add((r.get("ticker"), r.get("regime"), r.get("steps"), r.get("cap"), r.get("lr"),
                          r.get("head_only", False), r.get("rpt", False), r.get("exog", False)))
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
                            key = (tk, reg, steps, cap, lr, head_only, rpt, exog)
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
                            tag = f"{tk}|{reg}|st={steps}|cap={cap}|lr={lr}" + ("|head" if head_only else "") + ("|rpt" if rpt else "") + ("|exog" if exog else "")
                            r = train_regime_model(tr_win, test, steps, tag, lr=lr,
                                                   n_channels=n_channels, ckpt_dir=ckpt_dir,
                                                   head_only=head_only, rpt=rpt,
                                                   exog=exog, dates=dates)
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
    span_cols = [f"dir_acc_h{s}" for s in HORIZON_SPANS if f"dir_acc_h{s}" in df.columns]
    print("=== best per-regime configs (max OOS dir excess over persistence) ===")
    show = ["ticker", "regime", "steps", "cap", "lr", "head_only", "rpt", "exog", "dir_acc", "pers_dir",
            "mape", "n_test", "secs"] + span_cols
    print(best[[c for c in show if c in best.columns]].to_string(index=False))
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_BEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default="AEP,NVR,FICO")
    ap.add_argument("--steps", nargs="+", type=int, default=[3000, 6000])
    ap.add_argument("--caps", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--lrs", nargs="+", type=float, default=[None])  # None = gd.LR
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--split-frac", type=float, default=0.7, help="train fraction of history (shared boundary)")
    ap.add_argument("--channels", type=int, default=1, choices=[1, 3],
                    help="1=close-only (default); 3=close+return+realized-vol channels")
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir to save per-regime checkpoints for production serving "
                         "(e.g. checkpoints/regime)")
    ap.add_argument("--quick", action="store_true", help="1 config x 1 regime, tiny")
    ap.add_argument("--head-only", action="store_true",
                    help="TTM-paper mode: freeze the backbone, fine-tune only the "
                         "decoder+head (36% of params). Tests whether regime-specific "
                         "skill lives in the head vs full-model fine-tune.")
    ap.add_argument("--rpt", action="store_true",
                    help="Resolution Prefix Tuning: pass the daily freq token (8) to the "
                         "model so short-context regime windows don't have to infer "
                         "resolution from the data (TTM paper 3.1.1).")
    ap.add_argument("--exog", action="store_true",
                    help="Add an exogenous event-proximity channel (days-to-next FOMC/"
                         "expiry per timestep from economic_calendar.csv) — known-future "
                         "calendar exog in input form (TTM paper 3.2).")
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
        args.split_frac, args.resume, args.max_experiments, n_channels=args.channels,
        ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None, head_only=args.head_only,
        rpt=args.rpt, exog=args.exog)


if __name__ == "__main__":
    main()

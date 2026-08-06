"""pass3_sweep.py - Granite TTM parameter sweep (Pass 3).

Tests the model/objective axes the user asked about, all on the CLEANED
adjusted-history data (adj_close), using fixed200 windows (effective stride
~3, matching the production backfill) so windowing is held constant and only
the AXIS under test varies. Sample: AEP (low-vol), NVR (high-vol), FICO (mid).

Axes:
  context      : 256 / 512 / 1024
  horizon      : 32 / 96 / 240
  patch_length : 8 / 16 / 32
  use_decoder  : True / False
  lr           : 1e-5 / 1e-4 / 3e-4
  pretrained   : True (IBM warm) / False (from scratch)
  objective    : 'price' (raw adj_close) / 'returns' (log-returns; MAPE computed
                 by reconstructing price via cumsum from last known price)
  multivariate : False (price only) / True (+ normalized volume channel)
  ensemble     : 1 seed / 3 seeds averaged

Each experiment is one (axis, value) cell. We keep the OTHER axes at their
production defaults (context=512, horizon=96, patch=8, decoder=True, lr=1e-4,
pretrained=True, price, univariate, 1 seed) and vary ONE at a time, PLUS a few
combinatorial 'best-guess' cells.

Metrics: MAPE vs persistence, dir-accuracy, MAE (raw), all computed on the
held-in-sample windows (same as pass-1/2; this is a training-regime comparison,
not a strict walk-forward — noted as a limitation).
"""
import time, json, sys, copy
import numpy as np
import torch
import pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerConfig

device = b._device()
RAW = pd.read_parquet(b.PRICES)
# Clean the FULL frame ONCE (4.65M rows); all builders read from CLEAN.
CLEAN = _clean_price_frame(RAW, None)
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
from pass4 import BATCH, RECENT  # canonical training constants (was local)
TICKERS = ["AEP", "NVR", "FICO"]
STEPS = 6000  # comparable across cells
CAP = 500  # windows per ticker (linspace over available)


# ---------- data builders ----------
def cleaned_series(tk, objective="price"):
    cl = _clean_price_frame(RAW, None)
    s = cl[cl["ticker"] == tk]["close"].to_numpy().astype(float).ravel()
    s = s[np.isfinite(s)]
    if objective == "returns":
        # log-returns; target = future returns
        return np.log(s[1:] / s[:-1]).astype(np.float32), s[:-1]  # (ret, last_price_for_recon)
    return s.astype(np.float32), None


def _series_and_vol(tk, recent=RECENT):
    """Return (price_array, volume_array) aligned on the same valid rows.
    Reads from the module-level CLEAN frame (cleaned ONCE) for speed.
    Clips to the last `recent` trading days so the adjusted series is
    stationary (full adjusted history spans 0.5->138 = 270x, which breaks
    raw-value MSE training). 2520d ~= 10y of stationary adjusted history."""
    sub = CLEAN[CLEAN["ticker"] == tk]  # CLEAN is pre-sorted by (ticker, date)
    if recent is not None and recent > 0:
        sub = sub.tail(recent)
    price = sub["close"].to_numpy().astype(float).ravel()
    vol = sub["volume"].to_numpy().astype(float).ravel()
    mask = np.isfinite(price) & np.isfinite(vol)
    return price[mask], vol[mask]


def build_windows(tk, context, horizon, objective="price", multivariate=False):
    # Returns list of (ctx, tgt, extra) where ctx/tgt have shape (L, C):
    # sequence-first, channels-last (TTM wants (batch, seq, channels)).
    if objective == "returns":
        price, _ = _series_and_vol(tk)
        ret = np.log(price[1:] / price[:-1]).astype(np.float32)
        lastp = price[:-1]
        n = len(ret) - (context + horizon) + 1
        if n <= 0:
            return []
        idxs = np.linspace(0, n - 1, CAP).astype(int) if n > CAP else np.arange(n)
        wins = []
        for k in idxs:
            c = ret[k:k + context]; t = ret[k + context:k + context + horizon]
            if len(c) == context and len(t) == horizon:
                wins.append((c.reshape(context, 1).astype(np.float32),
                             t.reshape(horizon, 1).astype(np.float32),
                             float(lastp[k + context - 1])))
        return wins

    price, vol = _series_and_vol(tk)
    n = len(price) - (context + horizon) + 1
    if n <= 0:
        return []
    idxs = np.linspace(0, n - 1, CAP).astype(int) if n > CAP else np.arange(n)
    wins = []
    for k in idxs:
        c = price[k:k + context]; t = price[k + context:k + context + horizon]
        if len(c) != context or len(t) != horizon:
            continue
        if multivariate:
            vc = vol[k:k + context]; vt = vol[k + context:k + context + horizon]
            if len(vc) != context or len(vt) != horizon:
                continue
            cw = np.stack([c, _norm(vc)], axis=1).astype(np.float32)
            tw = np.stack([t, _norm(vt)], axis=1).astype(np.float32)
        else:
            cw = c.reshape(context, 1).astype(np.float32)
            tw = t.reshape(horizon, 1).astype(np.float32)
        wins.append((cw, tw, tk))
    return wins


def _norm(x):
    x = np.asarray(x, dtype=float)
    if x.std() < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


# ---------- model + train ----------
def make_model(context, horizon, patch_length, use_decoder, pretrained):
    # Load the PRETRAINED IBM config, then override only the swept fields so
    # the architecture matches the warm-start checkpoint (a bare
    # TinyTimeMixerConfig() builds a tiny d_model=16 arch that mismatches).
    # NOTE: the global `warm` ckpt was trained at IBM defaults
    # (context=512, horizon=96, patch=64, decoder=True). Any cell that changes
    # patch_length / use_decoder / horizon alters weight shapes, so `warm`
    # cannot load -> that cell MUST train from scratch (valid experiment:
    # does IBM pretrain help at this altered config?).
    CKPT_ARCH = dict(context=512, horizon=96, patch=64, decoder=True)
    arch_ok = (context == CKPT_ARCH["context"] and horizon == CKPT_ARCH["horizon"]
               and patch_length == CKPT_ARCH["patch"] and use_decoder == CKPT_ARCH["decoder"])
    if pretrained and not arch_ok:
        pretrained = False
    cfg = TinyTimeMixerConfig.from_pretrained(gd.DEFAULT_MODEL)
    cfg.context_length = context
    cfg.prediction_length = horizon
    cfg.patch_length = patch_length
    cfg.use_decoder = use_decoder
    m = TinyTimeMixerForPrediction(cfg)
    if pretrained:
        try:
            m.load_state_dict(warm, strict=False)
        except Exception as e:
            print(f"    warm-load partial: {e}", flush=True)
    return m.to(device)


def train_score(wins, context, horizon, patch_length, use_decoder, lr,
                pretrained, objective, seeds=1):
    # wins: list of (ctx, tgt, extra) where ctx/tgt shape (L, C)
    ctx = np.stack([w[0] for w in wins])           # (N, L, C)
    tgt = np.stack([w[1] for w in wins])           # (N, L, C)
    extra = [w[2] for w in wins]
    C = ctx.shape[2]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    preds_seeds = []
    for sd in range(seeds):
        torch.manual_seed(42 + sd)
        m = make_model(context, horizon, patch_length, use_decoder, pretrained)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        s = 0; t0 = time.time()
        while s < STEPS:
            for xb, yb in dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                o = m(past_values=xb, future_values=yb); loss = o.loss
                opt.zero_grad(); loss.backward(); opt.step(); s += 1
                if s >= STEPS: break
        m.eval()
        pa = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                out = m(past_values=xb)
                p = getattr(out, "prediction_outputs", out)
                if not isinstance(p, torch.Tensor):
                    p = p[0] if isinstance(p, (tuple, list)) else out
                pa.append(p.cpu().float().numpy())
        preds_seeds.append(np.concatenate(pa, 0))
    P = np.mean(preds_seeds, 0).squeeze()           # (N, H) for univariate
    A = tgt.squeeze()
    if objective == "returns":
        lastp = np.array([e for e in extra])        # last price before target
        cum = np.cumsum(P, axis=1)
        Pprice = lastp[:, None] * np.exp(cum)
        Acum = np.cumsum(A, axis=1)
        Aprice = lastp[:, None] * np.exp(Acum)
        P, A = Pprice, Aprice
    CL = ctx[:, -1, 0]                              # last known price (channel 0)
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    Pers = np.repeat(CL[:, None], horizon, axis=1)
    mape_pers = float((np.abs(Pers - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    a_dir = np.sign(A.mean(1) - CL); p_dir = np.sign(P.mean(1) - CL)
    dir_acc = float((a_dir == p_dir).mean() * 100)
    mae = float(np.abs(P - A).mean())
    return dict(mape=round(mape, 2), mape_pers=round(mape_pers, 2),
                dir_acc=round(dir_acc, 1), mae=round(mae, 3))


# ---------- experiment grid ----------
DEFAULTS = dict(context=512, horizon=96, patch_length=64, use_decoder=True,
                 lr=1e-4, pretrained=True, objective="price", multivariate=False, seeds=1)

GRID = [
    # baseline (IBM arch, warm-loaded)
    dict(name="baseline", **DEFAULTS),
    # horizon  (changes head dim -> from scratch)
    dict(name="hor32", horizon=32, pretrained=False), dict(name="hor240", horizon=240, pretrained=False),
    # decoder  (changes head dim -> from scratch)
    dict(name="no_decoder", use_decoder=False, pretrained=False),
    # lr       (arch_ok -> warm)
    dict(name="lr1e-5", lr=1e-5), dict(name="lr3e-4", lr=3e-4),
    # pretrained vs scratch (arch_ok)
    dict(name="scratch", pretrained=False),
    # objective (arch_ok -> warm)
    dict(name="returns", objective="returns"),
    # multivariate (+volume channel; arch_ok -> warm)
    dict(name="multi_vol", multivariate=True),
    # ensemble (3 seeds averaged)
    dict(name="ens3", seeds=3),
]
# NOTE: context_length and patch_length sweeps are NOT supported by TTM-R2 --
# its patch/feature-mixer sub-layer dims are frozen to IBM's config (512/64).
# Overriding them raises a matmul shape error, so they are omitted here.

rows = []
for tk in TICKERS:
    print(f"\n=== {tk} ===", flush=True)
    for exp in GRID:
        cfg = {**DEFAULTS, **exp}
        # skip multivariate/returns for horizon!=96 to limit combos? keep simple: run all
        wins = build_windows(tk, cfg["context"], cfg["horizon"], cfg["objective"], cfg["multivariate"])
        if not wins or len(wins) < 5:
            print(f"  [skip {exp['name']}] {len(wins)} windows", flush=True)
            continue
        try:
            r = train_score(wins, cfg["context"], cfg["horizon"], cfg["patch_length"],
                            cfg["use_decoder"], cfg["lr"], cfg["pretrained"],
                            cfg["objective"], cfg["seeds"])
        except Exception as e:
            print(f"  [ERR {exp['name']}] {type(e).__name__}: {e}", flush=True)
            continue
        r.update(ticker=tk, experiment=exp["name"], **{k: cfg[k] for k in
                ("context", "horizon", "patch_length", "use_decoder", "lr", "pretrained", "objective", "multivariate", "seeds")})
        rows.append(r)
        print(f"  {exp['name']:14} nw={len(wins):4} MAPE={r['mape']:6.2f}% "
              f"(pers={r['mape_pers']:.2f}%) dir={r['dir_acc']:5.1f}% MAE={r['mae']:.3f}", flush=True)

with open("/tmp/pass3.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\n=== PASS 3 DONE ===", flush=True)

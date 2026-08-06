"""pass4.py - Pass-4: re-run passes 2 & 3 on ADJUSTED closes, warm-started from
the freshly-trained ADJUSTED global checkpoint (train_adjusted_full.py output).

Now that we have proper adjusted checkpoints (matching the adjusted data
distribution), the warm-start is consistent and the passes are comparable:

Part A: PASS-3 PARAM GRID on ADJUSTED closes.
        Warm-load from the adjusted global ckpt (consistent). pretrained=False
        cells train from IBM scratch (still a valid from-scratch baseline).

Part B: PASS-2 REGIME SWEEP on ADJUSTED closes.
        Warm-started from the adjusted global ckpt (corrected: the old
        unadjusted-ckpt-on-adjusted-data mismatch gave 715% MAPE).

Both use adj_close via _clean_price_frame(use_adj=True).
"""
import time, json, numpy as np, torch, pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerConfig

device = b._device()
RAW = pd.read_parquet(b.PRICES)
ADJ_GLOBAL_DIR = gd.CKPT_DIR / "adjusted_global"
g_ckpt = gd.latest_ckpt_in(ADJ_GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
# Load the IBM base model ONCE (avoids repeated HF-hub fetches that stall/kill the run)
BASE_MODEL, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
BASE_MODEL = BASE_MODEL.to(device)
import copy
# Cache for variant-arch models (hor32/hor240/no_decoder) built once each
_VAR_CACHE = {}
def _variant_model(context, horizon, patch_length, use_decoder):
    key = (context, horizon, patch_length, use_decoder)
    if key not in _VAR_CACHE:
        cfg = TinyTimeMixerConfig.from_pretrained(gd.DEFAULT_MODEL)
        cfg.context_length = context; cfg.prediction_length = horizon
        cfg.patch_length = patch_length; cfg.use_decoder = use_decoder
        _VAR_CACHE[key] = TinyTimeMixerForPrediction(cfg).to(device)
    return _VAR_CACHE[key]
BATCH = 512
CONTEXT, HORIZON0 = gd.CONTEXT, gd.HORIZON
RECENT = 2520  # clip to last ~10y (stationary)

# ---- price source cache (one clean per use_adj) ----
_CLEAN = {}
def cleaned_price(tk, use_adj):
    key = "adj" if use_adj else "raw"
    if key not in _CLEAN:
        _CLEAN[key] = _clean_price_frame(RAW, RECENT, use_adj=use_adj)
    sub = _CLEAN[key][_CLEAN[key]["ticker"] == tk]
    return sub["close"].to_numpy().astype(float).ravel()

# =====================================================================
# PASS-3 grid
# =====================================================================
def build_windows_p3(tk, context, horizon, objective="price", multivariate=False, use_adj=True):
    if objective == "returns":
        price = cleaned_price(tk, use_adj)
        ret = np.log(price[1:] / price[:-1]).astype(np.float32)
        lastp = price[:-1]
        n = len(ret) - (context + horizon) + 1
        if n <= 0: return []
        idxs = np.linspace(0, n - 1, 200).astype(int)
        wins = []
        for k in idxs:
            c = ret[k:k + context]; t = ret[k + context:k + context + horizon]
            if len(c) == context and len(t) == horizon:
                wins.append((c.reshape(context, 1).astype(np.float32),
                             t.reshape(horizon, 1).astype(np.float32),
                             float(lastp[k + context - 1])))
        return wins
    price = cleaned_price(tk, use_adj)
    vol = cleaned_price_vol(tk, use_adj) if multivariate else None
    n = len(price) - (context + horizon) + 1
    if n <= 0: return []
    idxs = np.linspace(0, n - 1, 200).astype(int)
    wins = []
    for k in idxs:
        c = price[k:k + context]
        t = price[k + context:k + context + horizon]
        if len(c) == context and len(t) == horizon:
            if multivariate:
                vc = vol[k:k + context]; vt = vol[k + context:k + context + horizon]
                vc_n = (vc - vc.mean()) / (vc.std() + 1e-8); vt_n = (vt - vt.mean()) / (vt.std() + 1e-8)
                cw = np.stack([c, vc_n], axis=1).astype(np.float32)
                tw = np.stack([t, vt_n], axis=1).astype(np.float32)
            else:
                cw = c.reshape(context, 1).astype(np.float32)
                tw = t.reshape(horizon, 1).astype(np.float32)
            wins.append((cw, tw, float(price[k + context - 1])))
    return wins

def cleaned_price_vol(tk, use_adj):
    key = "adj" if use_adj else "raw"
    if key not in _CLEAN:
        _CLEAN[key] = _clean_price_frame(RAW, RECENT, use_adj=use_adj)
    sub = _CLEAN[key][_CLEAN[key]["ticker"] == tk]
    return sub["volume"].to_numpy().astype(float).ravel()

def make_model_p3(context, horizon, patch_length, use_decoder, pretrained):
    # Reuse the ONCE-loaded base model (deepcopy) to avoid per-cell HF fetches.
    if context == gd.CONTEXT and horizon == gd.HORIZON and patch_length == 64 and use_decoder is True:
        m = copy.deepcopy(BASE_MODEL)
    else:
        m = copy.deepcopy(_variant_model(context, horizon, patch_length, use_decoder))
    if pretrained:
        try:
            m.load_state_dict(warm, strict=False)
        except Exception as e:
            print(f"    [warm-load partial] {e}", flush=True)
    return m.to(device)

def train_score_p3(wins, context, horizon, patch, decoder, lr, pretrained, objective, seeds=1):
    ctx = np.stack([w[0] for w in wins]); tgt = np.stack([w[1] for w in wins])
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    # eval loader MUST be shuffle=False so predictions align row-for-row with tgt
    dl_eval = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                         batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=False)
    preds = []
    for sd in range(seeds):
        torch.manual_seed(42 + sd)
        m = make_model_p3(context, horizon, patch, decoder, pretrained).train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        s = 0
        while s < 6000:
            for xb, yb in dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                o = m(past_values=xb, future_values=yb); loss = o.loss
                if not torch.isfinite(loss):
                    print(f"    [NaN/inf loss, aborting seed {sd}]", flush=True); s = 6000; break
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step(); s += 1
                if s >= 6000: break
            if s >= 6000: break
        m.eval()
        with torch.no_grad():
            pa = []
            for xb, _ in dl_eval:
                out = m(past_values=xb.to(device))
                p = getattr(out, "prediction_outputs", out)
                if not isinstance(p, torch.Tensor):
                    p = p[0] if isinstance(p, (tuple, list)) else out
                pa.append(p.cpu().float().numpy())
            preds.append(np.concatenate(pa, 0))
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    P = np.mean(preds, 0)
    # multivariate predictions carry a channel dim (price, volume, ...); score
    # only the price channel (index 0) so shapes match the univariate targets.
    if P.ndim == 3:
        P = P[:, :, 0]
    P = P.squeeze()
    A = tgt.squeeze()
    if A.ndim == 3:
        A = A[:, :, 0]
    CL = ctx[:, -1, 0]
    if objective == "returns":
        # model predicts LOG-returns; reconstruct price LEVELS:
        # price_k = last_close * exp(cumsum(logret_0..k))
        P = CL[:, None] * np.exp(np.nan_to_num(P).cumsum(1))
        A = CL[:, None] * np.exp(np.nan_to_num(A).cumsum(1))
    P = np.nan_to_num(P); A = np.nan_to_num(A)
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    mape_p = float((np.abs(np.repeat(CL[:, None], horizon, 1) - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    dir_acc = float((np.sign(A.mean(1) - CL) == np.sign(P.mean(1) - CL)).mean() * 100)
    mae = float(np.abs(P - A).mean())
    return dict(mape=round(mape, 2), mape_pers=round(mape_p, 2), dir_acc=round(dir_acc, 1), mae=round(mae, 3))

P3_GRID = [
    dict(name="baseline", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-4, pretrained=True, objective="price", multivariate=False, seeds=1),
    dict(name="hor32", context=512, horizon=32, patch_length=64, use_decoder=True, lr=1e-4, pretrained=False, objective="price", multivariate=False, seeds=1),
    dict(name="hor240", context=512, horizon=240, patch_length=64, use_decoder=True, lr=1e-4, pretrained=False, objective="price", multivariate=False, seeds=1),
    dict(name="no_decoder", context=512, horizon=96, patch_length=64, use_decoder=False, lr=1e-4, pretrained=False, objective="price", multivariate=False, seeds=1),
    dict(name="lr1e-5", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-5, pretrained=True, objective="price", multivariate=False, seeds=1),
    dict(name="lr3e-4", context=512, horizon=96, patch_length=64, use_decoder=True, lr=3e-4, pretrained=True, objective="price", multivariate=False, seeds=1),
    dict(name="scratch", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-4, pretrained=False, objective="price", multivariate=False, seeds=1),
    dict(name="returns", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-4, pretrained=True, objective="returns", multivariate=False, seeds=1),
    dict(name="multi_vol", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-4, pretrained=True, objective="price", multivariate=True, seeds=1),
    dict(name="ens3", context=512, horizon=96, patch_length=64, use_decoder=True, lr=1e-4, pretrained=True, objective="price", multivariate=False, seeds=3),
]
# context/patch omitted: TTM-R2 patch/feature-mixer dims frozen to 512/64.

# =====================================================================
# PASS-2 regime sweep
# =====================================================================
def make_windows_p2(s, stride, cap):
    n = len(s) - (CONTEXT + HORIZON0) + 1
    idxs = np.arange(0, n, stride)
    if len(idxs) > cap:
        idxs = np.linspace(0, n - 1, cap).astype(int)
    out = []
    for k in idxs:
        c = s[k:k + CONTEXT]; t = s[k + CONTEXT:k + CONTEXT + HORIZON0]
        if len(c) == CONTEXT and len(t) == HORIZON0:
            out.append((c.astype(np.float32), t.astype(np.float32)))
    return out

def train_score_p2(wins, steps, warm_flag, tag):
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    dl_eval = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                         batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=False)
    m = copy.deepcopy(BASE_MODEL)
    if warm_flag:
        m.load_state_dict(warm, strict=False)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    s = 0; t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb); loss = o.loss
            if not torch.isfinite(loss):
                print(f"    [NaN/inf loss in {tag}, aborting]"); s = steps; break
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); s += 1
            if s >= steps: break
        if s >= steps: break
    dt = time.time() - t0
    m.eval()
    p_all, a_all, cl = [], [], []
    with torch.no_grad():
        for xb, yb in dl_eval:
            out = m(past_values=xb.to(device))
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            p_all.append(p.cpu().float().numpy()); a_all.append(yb.numpy())
            cl.append(xb[:, -1, 0].cpu().numpy())
    P = np.concatenate(p_all, 0).squeeze(-1); A = np.concatenate(a_all, 0).squeeze(-1)
    CL = np.concatenate(cl, 0)
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    mape_p = float((np.abs(np.repeat(CL[:, None], HORIZON0, 1) - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    dir_acc = float((np.sign(A.mean(1) - CL) == np.sign(P.mean(1) - CL)).mean() * 100)
    mae = float(np.abs(P - A).mean())
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(mape=round(mape, 2), mape_pers=round(mape_p, 2), dir_acc=round(dir_acc, 1),
                mae=round(mae, 3), n_windows=len(wins), secs=round(dt, 1), tag=tag)

P2_WIN = {
    "fixed200": dict(stride=1, cap=200),
    "scaled400": dict(stride=1, cap=400),
    "daily_stride1": dict(stride=1, cap=10_000_000),
    "half_wstride256": dict(stride=256, cap=10_000_000),
    "quarter_wstride128": dict(stride=128, cap=10_000_000),
}
P2_STEP = {"d6000": 6000, "d9000": 9000, "d12000": 12000}

# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    results = []
    print("########## PART A: PASS-3 GRID (ADJUSTED closes, warm=adjusted-ckpt) ##########", flush=True)
    for tk in ["AEP", "NVR", "FICO"]:
        print(f"\n=== P3 {tk} (adjusted) ===", flush=True)
        for c in P3_GRID:
            wins = build_windows_p3(tk, c["context"], c["horizon"], c["objective"], c["multivariate"], use_adj=True)
            r = train_score_p3(wins, c["context"], c["horizon"], c["patch_length"], c["use_decoder"],
                               c["lr"], c["pretrained"], c["objective"], c["seeds"])
            r.update(part="P3", ticker=tk, exp=c["name"], use_adj="adj")
            results.append(r)
            print(f"  {c['name']:12} nw={len(wins)} {r}", flush=True)

    print("\n########## PART B: PASS-2 REGIME (ADJUSTED closes, warm=adjusted-ckpt) ##########", flush=True)
    for tk in ["AEP", "NVR"]:
        s = cleaned_price(tk, use_adj=True)
        print(f"\n=== P2 {tk} (adjusted, n={len(s)}) ===", flush=True)
        for wname, wp in P2_WIN.items():
            wins = make_windows_p2(s, wp["stride"], wp["cap"])
            if len(wins) < 3:
                print(f"  [skip {wname}] only {len(wins)} windows", flush=True); continue
            for sname, steps in P2_STEP.items():
                tag = f"{tk}|{wname}|{sname}"
                r = train_score_p2(wins, steps, warm_flag=True, tag=tag)  # warm from adjusted global ckpt
                r.update(part="P2", ticker=tk, win=wname, step=sname, steps=steps, use_adj="adj")
                results.append(r)
                print(f"  {tag:30} nw={len(wins):4} steps={steps:6} MAPE={r['mape']:6.2f}% "
                      f"(pers={r['mape_pers']:.2f}%) dir={r['dir_acc']:5.1f}% MAE={r['mae']:7.3f} {r['secs']:.0f}s", flush=True)

    json.dump(results, open("/tmp/pass4_results.json", "w"), indent=2)
    print("\n=== PASS-4 DONE ===", flush=True)

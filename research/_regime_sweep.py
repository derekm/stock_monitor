import time, json, sys, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

RAW = pd.read_parquet(b.PRICES)
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
BATCH = 512

def cleaned_series(tk):
    s = b.build_full_history_windows(RAW, tickers=[tk], max_windows_per_ticker=10_000_000)
    # s is list of (ctx,tgt,tk); recover the raw cleaned close series from prices via _clean
    from granite_backfill import _clean_price_frame
    cl = _clean_price_frame(RAW, 2520)
    return cl[cl['ticker'] == tk]['close'].to_numpy().astype(np.float32)

def make_windows(s, stride, cap):
    n = len(s) - (CONTEXT + HORIZON) + 1
    idxs = np.arange(0, n, stride)
    if len(idxs) > cap:
        idxs = np.linspace(0, n - 1, cap).astype(int)
    out = []
    for k in idxs:
        c = s[k:k + CONTEXT]; t = s[k + CONTEXT:k + CONTEXT + HORIZON]
        if len(c) == CONTEXT and len(t) == HORIZON:
            out.append((c.astype(np.float32), t.astype(np.float32)))
    return out

def train_score(wins, steps, tag):
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    m = TinyTimeMixerForPrediction.from_pretrained(gd.DEFAULT_MODEL).to(device)
    m.load_state_dict(warm); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    s = 0; t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb); loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step(); s += 1
            if s >= steps: break
    dt = time.time() - t0
    # inference (no teacher forcing)
    m.eval()
    p_all, a_all, ctx_last = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            out = m(past_values=xb)
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            p_all.append(p.cpu().float().numpy()); a_all.append(yb.numpy())
            ctx_last.append(xb[:, -1, 0].cpu().numpy())
    P = np.concatenate(p_all, 0).squeeze(-1); A = np.concatenate(a_all, 0).squeeze(-1)
    CL = np.concatenate(ctx_last, 0)
    mae = float(np.abs(P - A).mean())
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    # persistence baseline (predict last context value flat)
    Pers = np.repeat(CL[:, None], HORIZON, axis=1)
    mape_pers = float((np.abs(Pers - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    # direction accuracy (sign of mean change over horizon)
    a_dir = np.sign(A.mean(1) - CL)
    p_dir = np.sign(P.mean(1) - CL)
    dir_acc = float((a_dir == p_dir).mean() * 100)
    return dict(mae=round(mae, 3), mape=round(mape, 2), mape_pers=round(mape_pers, 2),
                dir_acc=round(dir_acc, 1), n_windows=len(wins), secs=round(dt, 1), tag=tag)

WIN_REGIMES = {
    "fixed200":      dict(stride=1, cap=200),
    "scaled400":     dict(stride=1, cap=400),
    "daily_stride1": dict(stride=1, cap=10_000_000),
    "wstride512":    dict(stride=CONTEXT, cap=10_000_000),
}
STEP_REGIMES = {"low150": 150, "med600": 600, "high2000": 2000}

SMOKE = "--smoke" in sys.argv
TICKERS = ["AEP", "NVR"]
if SMOKE:
    TICKERS = ["AEP"]; STEP_REGIMES = {"low150": 20}

rows = []
for tk in TICKERS:
    s = cleaned_series(tk)
    print(f"\n=== {tk} (cleaned closes n={len(s)}) ===", flush=True)
    for wname, wp in WIN_REGIMES.items():
        wins = make_windows(s, wp['stride'], wp['cap'])
        nw = len(wins)
        for sname, steps in STEP_REGIMES.items():
            tag = f"{tk}|{wname}|{sname}"
            # scaled steps = epoch-matched vs fixed200 baseline
            scaled_steps = int(round(2000 * nw / 200)) if sname == "high2000" else steps
            actual_steps = scaled_steps if sname == "high2000" else steps
            r = train_score(wins, actual_steps, tag)
            r.update(ticker=tk, win=wname, step=sname, steps=actual_steps)
            rows.append(r)
            print(f"  {tag:28} nw={nw:4} steps={actual_steps:5} "
                  f"MAPE={r['mape']:6.2f}% (pers={r['mape_pers']:.2f}%) "
                  f"dir={r['dir_acc']:5.1f}% MAE={r['mae']:7.3f} {r['secs']:.0f}s", flush=True)

import csv
with open("/tmp/regime.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker","win","step","steps","n_windows","mape","mape_pers","dir_acc","mae","secs","tag"])
    w.writeheader(); w.writerows(rows)
print("\n=== REGIME SWEEP DONE ===", flush=True)

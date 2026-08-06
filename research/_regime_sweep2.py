import time, sys, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from granite_backfill import _clean_price_frame

RAW = pd.read_parquet(b.PRICES)
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
BATCH = 512

def cleaned_series(tk):
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
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    Pers = np.repeat(CL[:, None], HORIZON, axis=1)
    mape_pers = float((np.abs(Pers - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    a_dir = np.sign(A.mean(1) - CL); p_dir = np.sign(P.mean(1) - CL)
    dir_acc = float((a_dir == p_dir).mean() * 100)
    mae = float(np.abs(P - A).mean())
    return dict(mape=round(mape, 2), mape_pers=round(mape_pers, 2), dir_acc=round(dir_acc, 1),
                mae=round(mae, 3), n_windows=len(wins), secs=round(dt, 1), tag=tag)

# PASS 2: deeper steps + overlap strides
WIN_REGIMES = {
    "fixed200":      dict(stride=1, cap=200),
    "scaled400":     dict(stride=1, cap=400),
    "daily_stride1": dict(stride=1, cap=10_000_000),
    "half_wstride256": dict(stride=256, cap=10_000_000),   # ~6 windows (half-overlap of 512)
    "quarter_wstride128": dict(stride=128, cap=10_000_000), # ~14 windows
}
STEP_REGIMES = {"d6000": 6000, "d9000": 9000, "d12000": 12000}

# only run the two sample tickers from pass 1 for fair comparison
TICKERS = ["AEP", "NVR"]

rows = []
for tk in TICKERS:
    s = cleaned_series(tk)
    print(f"\n=== {tk} (cleaned closes n={len(s)}) ===", flush=True)
    for wname, wp in WIN_REGIMES.items():
        wins = make_windows(s, wp['stride'], wp['cap'])
        nw = len(wins)
        if nw < 3:
            print(f"  [skip {wname}] only {nw} windows (degenerate)", flush=True)
            continue
        for sname, steps in STEP_REGIMES.items():
            tag = f"{tk}|{wname}|{sname}"
            r = train_score(wins, steps, tag)
            r.update(ticker=tk, win=wname, step=sname, steps=steps)
            rows.append(r)
            print(f"  {tag:30} nw={nw:4} steps={steps:6} "
                  f"MAPE={r['mape']:6.2f}% (pers={r['mape_pers']:.2f}%) "
                  f"dir={r['dir_acc']:5.1f}% MAE={r['mae']:7.3f} {r['secs']:.0f}s", flush=True)

import csv
with open("/tmp/regime_pass2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker","win","step","steps","n_windows","mape","mape_pers","dir_acc","mae","secs","tag"])
    w.writeheader(); w.writerows(rows)
print("\n=== REGIME SWEEP PASS 2 DONE ===", flush=True)

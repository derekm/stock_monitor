import time, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from granite_backfill import score_windows

raw = pd.read_parquet(b.PRICES).drop_duplicates(subset=['ticker', 'date']).sort_values(['ticker', 'date'])
RECENT = 2520
raw = raw.groupby('ticker', group_keys=False).tail(RECENT)
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON

def make_windows(s, cap):
    n = len(s) - (CONTEXT + HORIZON) + 1
    idxs = np.arange(n)
    if len(idxs) > cap:
        idxs = np.linspace(0, n - 1, cap).astype(int)
    out = []
    for k in idxs:
        c = s[k:k+CONTEXT]; t = s[k+CONTEXT:k+CONTEXT+HORIZON]
        if len(c) == CONTEXT and len(t) == HORIZON:
            out.append((c.astype(np.float32), t.astype(np.float32), 'X'))
    return out

def train(wins, steps, batch=512):
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=batch, shuffle=True, pin_memory=True, drop_last=False)
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
    return round(score_windows(m, wins, device), 3), dt, len(wins)

for tk in ['FICO', 'NVR']:
    s = raw[raw['ticker'] == tk]['close'].to_numpy()
    if len(s) < CONTEXT + HORIZON:
        print(tk, 'too short'); continue
    nfull = len(s) - (CONTEXT + HORIZON) + 1
    print(f"\n=== {tk} (n={len(s)}, full_windows={nfull}) ===", flush=True)
    for cap, label in [(200, 'cap200'), (1_000_000, 'full(stride1)')]:
        wins = make_windows(s, cap)
        mae, dt, nw = train(wins, 2000)
        print(f"  [{label}] n_windows={nw}  MAE@2000={mae}  {dt:.0f}s", flush=True)
    # scale steps proportionally to window count to give full data equal epochs
    steps_scaled = int(2000 * nfull / 200)
    wins = make_windows(s, 1_000_000)
    mae_s, dt_s, _ = train(wins, steps_scaled)
    print(f"  [full stride1 @scaled={steps_scaled}] MAE={mae_s}  {dt_s:.0f}s", flush=True)
print("\n=== STRIDE TEST (CLEAN, FICO+NVR) DONE ===", flush=True)

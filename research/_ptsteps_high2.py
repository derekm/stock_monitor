import time, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
df = pd.read_parquet(b.PRICES)
wins = b.build_full_history_windows(df)
from collections import defaultdict
by = defaultdict(list)
for w in wins: by[w[2]].append(w)
# 2 easy + 2 volatile (volatile = the ones you care about for "more steps")
sample = ['F', 'NVR', 'AEP', 'FICO']
sample = [tk for tk in sample if tk in by]
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from granite_backfill import score_windows

BATCH = 512  # GPU-saturation config (proven 94% duty)

def train(tk, steps):
    ws = by[tk]
    ctx = np.stack([w[0] for w in ws])[:, :, None].astype(np.float32)
    tgt = np.stack([w[1] for w in ws])[:, :, None].astype(np.float32)
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    m = TinyTimeMixerForPrediction.from_pretrained(gd.DEFAULT_MODEL).to(device)
    m.load_state_dict(warm); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    s = 0
    t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb); loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step(); s += 1
            if s >= steps: break
    dt = time.time() - t0
    m.eval()
    mae = round(score_windows(m, ws, device), 3)
    return mae, dt

print(f"per-ticker 2000->9000 sweep, batch={BATCH} (GPU-saturated), 4 tickers", flush=True)
for tk in sample:
    row = f"{tk:5}"
    for steps in (2000, 4000, 6000, 9000):
        mae, dt = train(tk, steps)
        row += f"  {steps}:{mae:.3f}({dt:.0f}s)"
        print(row, flush=True)  # print progressively so we see completion
    print(row, flush=True)
print("=== SWEEP COMPLETE ===", flush=True)

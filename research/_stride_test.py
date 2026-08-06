import time, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from granite_backfill import build_full_history_windows, score_windows

df = pd.read_parquet(b.PRICES)
g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)

def train(wins, steps, batch=512):
    ctx = np.stack([w[0] for w in wins])[:, :, None].astype(np.float32)
    tgt = np.stack([w[1] for w in wins])[:, :, None].astype(np.float32)
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

for tk in ['AEP', 'FICO', 'NVR']:
    if tk not in df['ticker'].unique(): 
        print(tk, 'skip (no data)'); continue
    print(f"\n=== {tk} ===", flush=True)
    for cap, label in [(200, 'cap200(stride~22d)'), (1_000_000, 'full(stride1)')]:
        wins = build_full_history_windows(df, tickers=[tk], max_windows_per_ticker=cap)
        n = len(wins)
        mae, dt, _ = train(wins, 2000)
        print(f"  [{label}] n_windows={n}  MAE@{2000}={mae}  {dt:.0f}s", flush=True)
    # full stride-1 at higher steps — does more data keep helping?
    wins = build_full_history_windows(df, tickers=[tk], max_windows_per_ticker=1_000_000)
    mae6, dt6, _ = train(wins, 6000)
    print(f"  [full stride1 @6000] MAE={mae6}  {dt6:.0f}s", flush=True)
print("\n=== STRIDE TEST DONE ===", flush=True)

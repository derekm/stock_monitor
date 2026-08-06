import time, numpy as np, torch
import granite_backfill as b
from granite_backfill import score_windows, build_full_history_windows
gd = b.gd

device = b._device()
df = b.PRICES and __import__('pandas').read_parquet(b.PRICES)
wins = build_full_history_windows(df)
AEP = [w for w in wins if w[2] == 'AEP']
print(f"AEP windows: {len(AEP)}", flush=True)

g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
g_model, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
g_model = g_model.to(device)
g_model.load_state_dict(torch.load(g_ckpt, map_location=device))
base_mae = score_windows(g_model, AEP, device)
print(f"GLOBAL MAE (AEP): {base_mae:.4f}", flush=True)

# now train AEP for many steps from global warm start, like train_windows but deeper
from torch.utils.data import TensorDataset, DataLoader
ctx = np.stack([w[0] for w in AEP])[:, :, None]
tgt = np.stack([w[1] for w in AEP])[:, :, None]
ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
dl = DataLoader(ds, batch_size=64, shuffle=True, pin_memory=True)
m, _ = gd.load_granite_model(gd.DEFAULT_MODEL)
m = m.to(device)
m.load_state_dict(torch.load(g_ckpt, map_location=device))
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
for steps in (150, 600, 2000):
    t0 = time.time()
    s = 0
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb)
            loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step()
            s += 1
            if s >= steps:
                break
    mae = score_windows(m, AEP, device)
    print(f"  after {steps:5} steps: MAE={mae:.4f}  (global={base_mae:.4f}, Δ={mae-base_mae:+.4f}, {time.time()-t0:.1f}s)", flush=True)

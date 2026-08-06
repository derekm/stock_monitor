import time, glob, subprocess, psutil, numpy as np, torch
import granite_backfill as b

device = b._device()
import pandas as pd
df = pd.read_parquet(b.PRICES)
allw = b.build_full_history_windows(df)
g_ckpt = b.gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
model, _ = b.gd.load_granite_model(b.gd.DEFAULT_MODEL)
model = model.to(device)

def gpu():
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        return int(g.stdout.strip().split('\n')[0].replace(' %', ''))
    except Exception:
        return -1

# Train AEP, measuring GPU duty cycle per 0.1s tick across one full ticker
tk = 'AEP'
wins = [w for w in allw if w[2] == tk]

def gpu_trace(label, dur):
    samples = []
    t0 = time.time()
    while time.time() - t0 < dur:
        samples.append(gpu())
        time.sleep(0.1)
    busy = sum(1 for x in samples if x >= 60)
    print(f"  {label}: {dur}s window, gpu>=60% in {busy}/{len(samples)} ticks "
          f"({100*busy/len(samples):.0f}%)  min={min(samples)} max={max(samples)} mean={sum(samples)/len(samples):.0f}", flush=True)

# Before (glue): stack + load_state_dict
t0 = time.time()
model.load_state_dict(warm)
ctx = np.stack([w[0] for w in wins])[:, :, None]
tgt = np.stack([w[1] for w in wins])[:, :, None]
glue = time.time() - t0
print(f"GLUE (load_state_dict + np.stack): {glue*1000:.0f} ms", flush=True)

from torch.utils.data import TensorDataset, DataLoader
dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)), batch_size=64, shuffle=True)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=b.gd.LR)
step = 0
gpu_trace("DURING 150-step train", 8)
t0 = time.time()
while step < 150:
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        o = model(past_values=xb, future_values=yb)
        loss = o.loss
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step >= 150:
            break
train_t = time.time() - t0
# SAVE
t0 = time.time()
torch.save(model.state_dict(), '/tmp/_gap_ckpt.pt')
save_t = time.time() - t0
print(f"TRAIN: {train_t:.2f}s   SAVE: {save_t*1000:.0f} ms", flush=True)
print(f"Per-ticker GPU-idle fraction (glue+save)/(glue+train+save) = "
      f"{(glue+save_t)/(glue+train_t+save_t)*100:.1f}%", flush=True)

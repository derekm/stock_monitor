import time, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from granite_backfill import score_windows, build_full_history_windows

raw = pd.read_parquet(b.PRICES)
wins = build_full_history_windows(raw, tickers=['AEP'], max_windows_per_ticker=2000)
print(f"AEP windows for bench: {len(wins)}", flush=True)
ctx = np.stack([w[0] for w in wins])[:, :, None].astype(np.float32)
tgt = np.stack([w[1] for w in wins])[:, :, None].astype(np.float32)

g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON
BATCH = 512
STEPS = 2000

def make_model():
    m = TinyTimeMixerForPrediction.from_pretrained(gd.DEFAULT_MODEL).to(device)
    m.load_state_dict(warm); m.train()
    return m

def run(bench_mode):
    m = make_model()
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    if bench_mode == 'graph':
        # warmup + capture graph on a fixed sample
        static_in = [torch.empty(BATCH, CONTEXT, 1, device=device, dtype=torch.float32),
                     torch.empty(BATCH, HORIZON, 1, device=device, dtype=torch.float32)]
        s = 0
        # warmup steps
        for xb, yb in dl:
            xb = xb[:BATCH].to(device, non_blocking=True); yb = yb[:BATCH].to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb); loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step()
            s += 1
            if s >= 3: break
        # build static graph
        gx = torch.empty(BATCH, CONTEXT, 1, device=device); gy = torch.empty(BATCH, HORIZON, 1, device=device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(past_values=gx, future_values=gy)
            loss_g = out.loss
            opt.zero_grad(); loss_g.backward(); opt.step()
        t0 = time.time(); s = 0
        while s < STEPS:
            for xb, yb in dl:
                xb = xb[:BATCH].to(device, non_blocking=True); yb = yb[:BATCH].to(device, non_blocking=True)
                gx.copy_(xb); gy.copy_(yb)
                g.replay()
                s += 1
                if s >= STEPS: break
        dt = time.time() - t0
        m.eval()
        return round(score_windows(m, wins, device), 3), dt
    else:
        t0 = time.time(); s = 0
        while s < STEPS:
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                o = m(past_values=xb, future_values=yb); loss = o.loss
                opt.zero_grad(); loss.backward(); opt.step()
                s += 1
                if s >= STEPS: break
        dt = time.time() - t0
        m.eval()
        return round(score_windows(m, wins, device), 3), dt

for mode in ['eager', 'graph']:
    mae, dt = run(mode)
    print(f"  [{mode}] MAE={mae}  {STEPS} steps @batch{BATCH}  {dt:.1f}s  ({STEPS/dt:.1f} steps/s)", flush=True)
print("\n=== CUDA GRAPH BENCH DONE ===", flush=True)

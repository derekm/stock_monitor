import time, os, glob, json, copy, threading, subprocess, numpy as np, torch
import granite_backfill as b
gd = b.gd
device = b._device()

import pandas as pd
df = pd.read_parquet(b.PRICES)
wins = b.build_full_history_windows(df)
from collections import defaultdict
by = defaultdict(list)
for w in wins: by[w[2]].append(w)
# pick 6 tickers each with ~200 windows (the common case) for a fair comparison
cands = sorted([(tk, len(ws)) for tk, ws in by.items() if len(ws) == 200])
picked = [tk for tk, _ in cands[:6]]
print("picked tickers:", picked, flush=True)

g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)

def load_base():
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    m = TinyTimeMixerForPrediction.from_pretrained(gd.DEFAULT_MODEL)
    return m.to(device)

def train_one(model, tk, steps):
    ws = by[tk]
    ctx = np.stack([w[0] for w in ws])[:, :, None]
    tgt = np.stack([w[1] for w in ws])[:, :, None]
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=64, shuffle=True, pin_memory=True)
    model.load_state_dict(warm)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=gd.LR)
    s = 0
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = model(past_values=xb, future_values=yb)
            loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step()
            s += 1
            if s >= steps: break
    return float(loss.item())

def gpu_watch(dur, out):
    t0 = time.time()
    while time.time() - t0 < dur:
        try:
            g = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],
                                capture_output=True, text=True)
            out.append(int(g.stdout.strip().split('\n')[0].replace(' %','')))
        except Exception: pass
        time.sleep(0.2)

STEPS = 600

# ---- SERIAL ----
base = load_base()
t0 = time.time(); gpu = []
th = threading.Thread(target=gpu_watch, args=(max(40, STEPS*0.05*6), gpu)); th.start()
for tk in picked:
    train_one(base, tk, STEPS)
serial_t = time.time() - t0; th.join()
serial_gpu = sum(1 for x in gpu if x >= 50) / max(1, len(gpu))
print(f"SERIAL {len(picked)} tickers x{STEPS}: {serial_t:.1f}s ({serial_t/len(picked):.1f}s/tk) "
      f"gpu>=50%={100*serial_gpu:.0f}%", flush=True)
del base

# ---- PARALLEL T ----
def run_parallel(T):
    base0 = load_base()  # materialized on main thread, then deepcopy per worker
    queue = list(picked)
    lock = threading.Lock()
    results = {}
    gpu = []
    def worker(wid):
        m = copy.deepcopy(base0)  # independent instance -> thread-safe
        while True:
            with lock:
                if not queue: break
                tk = queue.pop(0)
            try:
                results[tk] = train_one(m, tk, STEPS)
            except Exception as e:
                results[tk] = f"ERR {e}"
    t0 = time.time()
    thw = threading.Thread(target=gpu_watch, args=(max(40, STEPS*0.05*len(picked)/T), gpu)); thw.start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(T)]
    for t in threads: t.start()
    for t in threads: t.join()
    thw.join()
    dt = time.time() - t0
    gg = sum(1 for x in gpu if x >= 50) / max(1, len(gpu))
    print(f"PARALLEL T={T} {len(picked)} tickers x{STEPS}: {dt:.1f}s ({dt/len(picked):.1f}s/tk) "
          f"gpu>=50%={100*gg:.0f}%  speedup_vs_serial={serial_t/dt:.2f}x", flush=True)
    del base0
    return dt

for T in (2, 3):
    run_parallel(T)

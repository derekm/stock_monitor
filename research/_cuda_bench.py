import time, numpy as np, torch, threading, subprocess, copy
import granite_backfill as b
gd = b.gd
device = b._device()
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

df = pd.read_parquet(b.PRICES)
wins = b.build_full_history_windows(df)
AEP = [w for w in wins if w[2] == 'AEP']
# build a 1024-sample dataset by repeating AEP windows (enough for batch=512)
rep = (AEP * 6)[:1024]
ctx = np.stack([w[0] for w in rep])[:, :, None].astype(np.float32)
tgt = np.stack([w[1] for w in rep])[:, :, None].astype(np.float32)
full_ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
print(f"bench dataset: {len(full_ds)} samples, x shape {ctx.shape}", flush=True)

g_ckpt = gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)

def gpu_watch(dur, out):
    t0 = time.time()
    while time.time() - t0 < dur:
        try:
            g = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],
                                capture_output=True, text=True)
            out.append(int(g.stdout.strip().split('\n')[0].replace(' %','')))
        except Exception:
            pass
        time.sleep(0.15)

STEPS = 2000

def run_cfg(name, batch, compile_model, use_graph):
    m = TinyTimeMixerForPrediction.from_pretrained(gd.DEFAULT_MODEL).to(device)
    m.load_state_dict(warm)
    if compile_model:
        t0 = time.time()
        m = torch.compile(m)
        print(f"  [{name}] torch.compile took {time.time()-t0:.1f}s", flush=True)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    dl = DataLoader(full_ds, batch_size=batch, shuffle=True, pin_memory=True)
    gpu = []
    # ---- CUDA graph path ----
    if use_graph:
        static_x = torch.empty(batch, ctx.shape[1], 1, device=device, dtype=torch.float32)
        static_y = torch.empty(batch, tgt.shape[1], 1, device=device, dtype=torch.float32)
        it = iter(dl)
        def next_batch():
            try: return next(it)
            except StopIteration: return None
        # warmup
        for _ in range(4):
            xb, yb = next_batch() or next(iter(dl))
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            static_x.copy_(xb); static_y.copy_(yb)
            out = m(past_values=static_x, future_values=static_y); out.loss.backward(); opt.step(); opt.zero_grad()
        g = torch.cuda.CudaGraph()
        with torch.cuda.graph(g):
            out = m(past_values=static_x, future_values=static_y)
            loss = out.loss
            loss.backward()
        # timed run
        t0 = time.time()
        th = threading.Thread(target=gpu_watch, args=(max(20, STEPS*0.01), gpu)); th.start()
        for s in range(STEPS):
            xb, yb = next_batch() or next(iter(dl))
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            static_x.copy_(xb); static_y.copy_(yb)
            g.replay()
            opt.step(); opt.zero_grad()
        th.join()
        dt = time.time() - t0
    else:
        t0 = time.time()
        th = threading.Thread(target=gpu_watch, args=(max(20, STEPS*0.01), gpu)); th.start()
        s = 0
        while s < STEPS:
            for xb, yb in dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                out = m(past_values=xb, future_values=yb); loss = out.loss
                opt.zero_grad(); loss.backward(); opt.step()
                s += 1
                if s >= STEPS: break
        th.join()
        dt = time.time() - t0
    duty = sum(1 for x in gpu if x >= 50) / max(1, len(gpu))
    print(f"  [{name}] batch={batch} steps={STEPS}: {dt:.1f}s  gpu>=50%={100*duty:.0f}%  "
          f"mean={sum(gpu)/max(1,len(gpu)):.0f}% max={max(gpu)}", flush=True)
    del m

print("=== CUDA saturation benchmark (2000 steps, AEP x6) ===", flush=True)
run_cfg("baseline", batch=64, compile_model=False, use_graph=False)
run_cfg("batch512", batch=512, compile_model=False, use_graph=False)
run_cfg("batch512+compile", batch=512, compile_model=True, use_graph=False)
try:
    run_cfg("batch512+compile+graph", batch=512, compile_model=True, use_graph=True)
except Exception as e:
    print(f"  [batch512+compile+graph] FAILED: {type(e).__name__}: {e}", flush=True)
print("=== done ===", flush=True)

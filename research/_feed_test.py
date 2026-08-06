import time, glob, subprocess, psutil, numpy as np, torch, threading
import granite_backfill as b

device = b._device()
import pandas as pd
df = pd.read_parquet(b.PRICES)
allw = b.build_full_history_windows(df)
tk = 'AEP'
wins = [w for w in allw if w[2] == tk]
g_ckpt = b.gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)
model, _ = b.gd.load_granite_model(b.gd.DEFAULT_MODEL)
model = model.to(device)
model.load_state_dict(warm)

def gpu_poll(secs, out):
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                '--format=csv,noheader'], capture_output=True, text=True)
            out.append(int(g.stdout.strip().split('\n')[0].replace(' %', '')))
        except Exception:
            pass
        time.sleep(0.1)

def run(cfg_name, num_workers, pin, nonblock):
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=64, shuffle=True,
                    num_workers=num_workers, pin_memory=pin)
    model.load_state_dict(warm)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=b.gd.LR)
    out = []
    th = threading.Thread(target=gpu_poll, args=(8, out)); th.start()
    t0 = time.time(); step = 0
    while step < 150:
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=nonblock)
            yb = yb.to(device, non_blocking=nonblock)
            o = model(past_values=xb, future_values=yb)
            loss = o.loss
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step >= 150:
                break
    th.join(); dt = time.time() - t0
    busy = sum(1 for x in out if x >= 60)
    print(f"{cfg_name}: {dt:.2f}s  gpu>=60%: {100*busy/len(out):.0f}%  "
          f"mean={sum(out)/len(out):.0f}%  max={max(out)}%", flush=True)

run("baseline (nw=0, pin=False)", 0, False, False)
run("pin_memory+nonblock (nw=0)", 0, True, True)
run("num_workers=2+pin+nonblock", 2, True, True)

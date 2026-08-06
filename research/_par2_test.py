import time, numpy as np, torch, threading, subprocess, traceback
import granite_backfill as b
device = b._device()
import pandas as pd
df = pd.read_parquet(b.PRICES)
allw = b.build_full_history_windows(df)
g_ckpt = b.gd.latest_ckpt_in(b.GLOBAL_DIR)
warm = torch.load(g_ckpt, map_location=device)

def gpu_poll(secs, out):
    t0 = time.time()
    while time.time() - t0 < secs:
        g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True)
        try:
            out.append(int(g.stdout.strip().split('\n')[0].replace(' %', '')))
        except Exception:
            pass
        time.sleep(0.2)

def train_one(tk):
    try:
        wins = [w for w in allw if w[2] == tk]
        model, _ = b.gd.load_granite_model(b.gd.DEFAULT_MODEL)
        model = model.to(device)
        model.load_state_dict(warm)
        from torch.utils.data import TensorDataset, DataLoader
        ctx = np.stack([w[0] for w in wins])[:, :, None]
        tgt = np.stack([w[1] for w in wins])[:, :, None]
        dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                        batch_size=64, shuffle=True)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=b.gd.LR)
        step = 0
        while step < 150:
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                o = model(past_values=xb, future_values=yb)
                loss = o.loss
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
                if step >= 150:
                    break
    except Exception:
        traceback.print_exc()

out = []; th = threading.Thread(target=gpu_poll, args=(16, out)); th.start()
t0 = time.time()
a = threading.Thread(target=train_one, args=('AEP',)); c = threading.Thread(target=train_one, args=('BA',))
a.start(); c.start(); a.join(); c.join()
par = time.time() - t0; th.join()
print(f'PARALLEL 2 tickers: {par:.1f}s  gpu_mean={sum(out)/len(out):.0f}%  speedup={27.3/par:.2f}x', flush=True)

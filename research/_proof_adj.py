import numpy as np, torch, time, pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd, train_windows
import granite_daily as gd2
device = b._device()
RAW = pd.read_parquet(b.PRICES)

def windows_stride1(tk, use_adj):
    df = _clean_price_frame(RAW, None, use_adj=use_adj)
    s = df[df["ticker"]==tk]["close"].to_numpy().astype(np.float32).ravel()
    C,H = gd.CONTEXT, gd.HORIZON
    n = len(s)-C-H+1
    idxs = np.arange(n)
    if len(idxs) > 4000:
        idxs = np.linspace(0, n-1, 4000).astype(int)
    return [(s[k:k+C], s[k+C:k+C+H]) for k in idxs if len(s[k:k+C])==C and len(s[k+C:k+C+H])==H]

def mape(model, wins):
    model.eval(); P=[]; A=[]
    with torch.no_grad():
        for c,t in wins:
            xb=torch.tensor(c[:,None], dtype=torch.float32)[None].to(device)
            o=model(past_values=xb); p=getattr(o,'prediction_outputs',o)
            if not isinstance(p,torch.Tensor): p=p[0] if isinstance(p,(tuple,list)) else o
            P.append(p[0].cpu().float().numpy()); A.append(t)
    P=np.array(P).squeeze(); A=np.array(A).squeeze()
    return float((np.abs(P-A)/np.abs(A).clip(min=1e-6)).mean()*100)

for tk in ["AEP","NVR"]:
    df = _clean_price_frame(RAW, None, use_adj=True)
    s = df[df["ticker"]==tk]["close"].to_numpy().astype(np.float32).ravel()
    wins = windows_stride1(tk, True)
    print(f"{tk}: n_closes={len(s)} n_windows(stride1,cap4000)={len(wins)}", flush=True)
    out = b.CKPT_DIR / "_proof" / tk
    out.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    ck = train_windows(wins, 12000, device, out, name="proof", batch=64)
    m = gd2.load_granite_model(gd.DEFAULT_MODEL)[0].to(device)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    print(f"{tk}: trained 12000 steps on {len(wins)} adj windows -> MAPE={mape(m,wins):.2f}% ({time.time()-t0:.0f}s)", flush=True)
print("PROOF DONE", flush=True)

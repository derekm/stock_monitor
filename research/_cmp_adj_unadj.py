import numpy as np, torch, time, pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd, train_windows
import granite_daily as gd2
device = b._device()
RAW = pd.read_parquet(b.PRICES)

def series(tk, use_adj):
    df = _clean_price_frame(RAW, None, use_adj=use_adj)
    s = df[df["ticker"]==tk]["close"].to_numpy().astype(float).ravel()
    return s

def windows(s, n=200):
    C, H = gd.CONTEXT, gd.HORIZON
    idxs = np.linspace(0, len(s)-C-H, n).astype(int)
    return [(s[k:k+C].astype(np.float32), s[k+C:k+C+H].astype(np.float32)) for k in idxs if len(s[k:k+C])==C and len(s[k+C:k+C+H])==H]

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

def train_ckpt(tk, use_adj, steps=150, nw=200):
    s = series(tk, use_adj)
    wins = windows(s, nw)
    out = b.PER_TICKER_DIR.parent / f"_cmp_{tk}_{'adj' if use_adj else 'unadj'}"
    out.mkdir(parents=True, exist_ok=True)
    ck = train_windows(wins, steps, device, out, name="cmp", batch=16)
    m = gd2.load_granite_model(gd.DEFAULT_MODEL)[0].to(device)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    return mape(m, wins)

for tk in ["AEP","NVR"]:
    for use_adj in (False, True):
        s = series(tk, use_adj)
        wins = windows(s, 200)
        ibm,_ = gd2.load_granite_model(gd.DEFAULT_MODEL)
        ibm_mape = mape(ibm.to(device), wins)
        m = train_ckpt(tk, use_adj, 150, 200)
        print(f"{tk} {'adj' if use_adj else 'unadj':5} n={len(s):5} ibm_zero_shot={ibm_mape:6.2f}%  trained150={m:6.2f}%", flush=True)

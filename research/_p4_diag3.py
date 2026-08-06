import numpy as np, torch
import pass4 as P
import granite_backfill as b
import granite_daily as gd
from granite_backfill import _clean_price_frame
device = P.device

# Build AEP adjusted windows in the SAME format train_aggregate/score_windows use: (c, t, tk)
df = __import__("pandas").read_parquet(b.PRICES)
clean = _clean_price_frame(df, recent_trading_days=P.RECENT, use_adj=True)
s = clean[clean["ticker"]=="AEP"]["close"].to_numpy().astype(np.float32)
wins=[]
n=len(s)-(512+96)+1
idxs=np.linspace(0,n-1,2000).astype(int)
for k in idxs:
    c=s[k:k+512]; t=s[k+512:k+512+96]
    if len(c)==512 and len(t)==96: wins.append((c,t,"AEP"))
print("AEP adj windows", len(wins), flush=True)

# Trusted model load
model, kind = gd.load_granite_model(gd.DEFAULT_MODEL)
model = model.to(device)
sd = torch.load(gd.latest_ckpt_in(gd.CKPT_DIR/"adjusted_global"), map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
mae = b.score_windows(model, wins, device)
print(f"TRUSTED score_windows (MAE) AEP adj: {mae:.4f}", flush=True)
# also raw MAPE via trusted windows
import torch as T
from torch.utils.data import TensorDataset, DataLoader
ctx=np.stack([w[0] for w in wins])[:,:,None]; tgt=np.stack([w[1] for w in wins])[:,:,None]
dl=DataLoader(TensorDataset(T.tensor(ctx),T.tensor(tgt)),batch_size=512,shuffle=False)
pa=[]
with torch.no_grad():
    for xb,_ in dl:
        o=model(past_values=xb.to(device)); p=getattr(o,'prediction_outputs',o)
        if not isinstance(p,T.Tensor): p=p[0] if isinstance(p,(tuple,list)) else o
        pa.append(p.cpu().float().numpy())
Pp=np.concatenate(pa,0); A=tgt
print("Pp shape",Pp.shape,"A shape",A.shape, flush=True)
# squeeze carefully
Pp2=Pp.squeeze(); A2=A.squeeze()
print("Pp2",Pp2.shape,"A2",A2.shape, flush=True)
mape=float((np.abs(Pp2-A2)/np.abs(A2).clip(min=1e-6)).mean()*100)
print(f"MAPE careful squeeze: {mape:.2f}%", flush=True)

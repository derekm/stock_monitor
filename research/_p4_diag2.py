import time, numpy as np, torch
import pass4 as P
import granite_daily as gd
from granite_backfill import gd as b_gd
device = P.device

# Build AEP adjusted windows (2000, same as pass4 baseline)
wins = P.build_windows_p3('AEP', 512, 96, 'price', False, use_adj=True)
ctx = np.stack([w[0] for w in wins]); tgt = np.stack([w[1] for w in wins])
from torch.utils.data import TensorDataset, DataLoader
dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)), batch_size=512, shuffle=True, pin_memory=True)

# METHOD A: gd.load_granite_model (exact arch) + adjusted warm  -> baseline-correct
modelA, kind = gd.load_granite_model(gd.DEFAULT_MODEL)
modelA = modelA.to(device)
adj_ckpt = gd.latest_ckpt_in(gd.CKPT_DIR / "adjusted_global")
sd = torch.load(adj_ckpt, map_location=device)
miss = modelA.load_state_dict(sd, strict=False)
modelA.eval()
print("METHOD A warm load: missing/unexpected:", miss, flush=True)
def mape_of(model):
    model.eval(); pa=[]
    with torch.no_grad():
        for xb,_ in dl:
            o=model(past_values=xb.to(device)); p=getattr(o,'prediction_outputs',o)
            if not isinstance(p,torch.Tensor): p=p[0] if isinstance(p,(tuple,list)) else o
            pa.append(p.cpu().float().numpy())
    Pp=np.concatenate(pa,0).squeeze(); A=tgt.squeeze()
    return float((np.abs(Pp-A)/np.abs(A).clip(min=1e-6)).mean()*100)
print(f"METHOD A warm-only MAPE = {mape_of(modelA):.2f}%", flush=True)

# Train METHOD A 6000 steps, log MAPE at checkpoints
opt = torch.optim.AdamW(modelA.parameters(), lr=1e-4)
modelA.train()
s=0; t0=time.time()
for xb,yb in dl:
    xb,yb=xb.to(device),yb.to(device)
    o=modelA(past_values=xb,future_values=yb); loss=o.loss
    opt.zero_grad(); loss.backward(); opt.step(); s+=1
    if s in (1,10,100,1000,6000):
        print(f"  A step {s} loss {float(loss):.4f} mape {mape_of(modelA):.2f}%", flush=True)
    if s>=6000: break
print(f"METHOD A done {time.time()-t0:.1f}s", flush=True)

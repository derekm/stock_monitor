import time, numpy as np, torch
import pass4 as P
device = P.device
wins = P.build_windows_p3('AEP', 512, 96, 'price', False, use_adj=False)
print('AEP unadj wins', len(wins), flush=True)
ctx = np.stack([w[0] for w in wins]); tgt = np.stack([w[1] for w in wins])
from torch.utils.data import TensorDataset, DataLoader
dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)), batch_size=512, shuffle=True, pin_memory=True)
m = P.make_model_p3(512, 96, 64, True, True).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
# log loss at 0,50,100,200 steps and MAPE
def mape_now():
    m.eval(); pa=[]
    with torch.no_grad():
        for xb,_ in dl:
            o=m(past_values=xb.to(device)); p=getattr(o,'prediction_outputs',o)
            if not isinstance(p,torch.Tensor): p=p[0] if isinstance(p,(tuple,list)) else o
            pa.append(p.cpu().float().numpy())
    Pp=np.concatenate(pa,0).squeeze(); A=tgt.squeeze(); CL=ctx[:,-1,0]
    return float((np.abs(Pp-A)/np.abs(A).clip(min=1e-6)).mean()*100)
s=0
for xb,yb in dl:
    xb,yb=xb.to(device),yb.to(device)
    o=m(past_values=xb,future_values=yb); loss=o.loss
    opt.zero_grad(); loss.backward(); opt.step(); s+=1
    if s in (1,10,50,100,200):
        print(f'step {s} loss {float(loss):.4f} mape {mape_now():.2f}%', flush=True)
    if s>=200: break
print('DONE', flush=True)

import torch, numpy as np, time
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd
from torch.utils.data import TensorDataset, DataLoader
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerConfig

device=b._device()
import pandas as pd
RAW=pd.read_parquet(b.PRICES)
clean=_clean_price_frame(RAW, None)
aep=clean[clean['ticker']=='AEP']['close'].to_numpy().astype(float).ravel()
print('AEP adjusted close: n=',len(aep),'min',round(aep.min(),3),'max',round(aep.max(),3),'mean',round(aep.mean(),2))
print('first 5:', np.round(aep[:5],3), 'last 5:', np.round(aep[-5],3))
# build 200 fixed windows (linspace) context=512
ctx_len,hor=512,96
n=len(aep)-(ctx_len+hor)+1
idxs=np.linspace(0,n-1,200).astype(int)
ctx=np.stack([aep[k:k+ctx_len] for k in idxs])[:,:,None].astype(np.float32)
tgt=np.stack([aep[k+ctx_len:k+ctx_len+hor] for k in idxs])[:,:,None].astype(np.float32)
print('ctx range per window sample0: min',round(ctx[0].min(),3),'max',round(ctx[0].max(),3))
print('tgt sample0 min/max:',round(tgt[0].min(),3),round(tgt[0].max(),3))
dl=DataLoader(TensorDataset(torch.tensor(ctx),torch.tensor(tgt)),batch_size=512,shuffle=True,pin_memory=True)
cfg=TinyTimeMixerConfig.from_pretrained(b.gd.DEFAULT_MODEL); cfg.context_length=512;cfg.prediction_length=96;cfg.patch_length=64;cfg.use_decoder=True
m=TinyTimeMixerForPrediction(cfg).to(device)
m.load_state_dict(torch.load(b.gd.latest_ckpt_in(b.gd.GLOBAL_DIR),map_location=device),strict=False)
m.train()
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
print('training 200 steps...',flush=True)
t0=time.time(); s=0
while s<200:
    for xb,yb in dl:
        xb,yb=xb.to(device,non_blocking=True),yb.to(device,non_blocking=True)
        o=m(past_values=xb,future_values=yb); loss=o.loss
        opt.zero_grad();loss.backward();opt.step();s+=1
        if s>=200:break
print('200 steps in',round(time.time()-t0,1),'s, final loss',round(float(loss),4),flush=True)
m.eval()
with torch.no_grad():
    out=m(past_values=torch.tensor(ctx[:3]).to(device))
    p=getattr(out,'prediction_outputs',out)
    if not isinstance(p,torch.Tensor): p=p[0] if isinstance(p,(tuple,list)) else out
    p=p.cpu().float().numpy()
print('PRED sample0 shape',p[0].shape)
print('PRED sample0 first10:',np.round(p[0,:10].ravel(),3))
print('TGT  sample0 first10:',np.round(tgt[0,:10].ravel(),3))
print('PRED sample0 last10:',np.round(p[0,-10:].ravel(),3))
print('TGT  sample0 last10:',np.round(tgt[0,-10:].ravel(),3))

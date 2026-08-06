import torch, numpy as np
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerConfig
import granite_backfill as b
device=b._device()
warm=torch.load(b.gd.latest_ckpt_in(b.GLOBAL_DIR), map_location=device)
# build exactly like make_model baseline
cfg=TinyTimeMixerConfig.from_pretrained(b.gd.DEFAULT_MODEL)
cfg.context_length=512; cfg.prediction_length=96; cfg.patch_length=64; cfg.use_decoder=True
print('cfg context', cfg.context_length, 'patch', cfg.patch_length)
m=TinyTimeMixerForPrediction(cfg).to(device)
print('model.config.context_length', m.config.context_length)
sd=m.load_state_dict(warm, strict=False)
print('missing', len(sd.missing_keys), 'unexpected', len(sd.unexpected_keys))
# single forward with a 512 window
xb=torch.randn(1,1,512,device=device)
try:
    out=m(past_values=xb)
    print('FWD OK pred', getattr(out,'prediction_outputs',out).shape)
except Exception as e:
    print('FWD ERR', repr(e))

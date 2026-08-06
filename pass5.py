"""pass5.py - OUT-OF-SAMPLE (honest) Granite-TTM evaluation.

pass4 measured IN-SAMPLE error: it built windows across the whole history and
trained + scored on the SAME windows, warm-starting from a checkpoint trained on
all history. That is memorization, not forecasting. Neural TTM models cannot be
"backtested" on data they trained on.

Two honest evaluation protocols (both temporally disjoint -> no leakage):

  MODE trainlast (DEFAULT): train on the LAST 10y (what production uses,
  RECENT=2520), test by forecasting the ~10y IMMEDIATELY BEFORE that window.
  Train and test regions are disjoint and separated by a gap, so the model
  never sees test prices. Tests the real production training regime.

  MODE half: train on the FIRST half of history, test on the SECOND half
  (expanding-origin style).

Honesty rules baked in:
  * Trained from the IBM base model only (pretrained=False) -> no full-history
    checkpoint contamination of the holdout.
  * Persistence baseline computed on the SAME test windows (apples-to-apples).
  * Test windows are entirely within the held-out region (no straddle leakage).

Usage:
    python pass5.py                              # trainlast, AEP/NVR/FICO, steps=6000
    python pass5.py --mode half --cutoff-frac 0.5
    python pass5.py --tickers AEP KO XOM --steps 9000 --strides fixed200 scaled400
Results -> /tmp/pass5_results.json + printed to stdout.
"""
import argparse, time, json, copy
import numpy as np, torch, pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame, gd
from torch.utils.data import TensorDataset, DataLoader
import pass4  # reuse model builders / cleaned_price / device / BATCH / BASE_MODEL / warm

device = pass4.device
BATCH = pass4.BATCH
BASE_MODEL = pass4.BASE_MODEL
CONTEXT, HORIZON = gd.CONTEXT, gd.HORIZON  # 512, 96
TRAIN_LEN = 2520   # ~10y, matches production RECENT
TEST_LEN = 2520    # ~10y preceding the train window


def clean_series(tk, use_adj=True):
    # full history (NO RECENT clip) so the split is real
    df = _clean_price_frame(pass4.RAW, 10_000_000, use_adj=use_adj)
    sub = df[df["ticker"] == tk]
    return sub["close"].to_numpy().astype(float).ravel()


def _windows_in_block(s, lo, hi, stride, cap):
    """Windows whose [context..target] lie entirely within [lo, hi)."""
    n = len(s)
    out = []
    max_k = hi - (CONTEXT + HORIZON)
    if max_k < lo:
        return out
    idxs = np.arange(lo, max_k + 1, stride)
    if len(idxs) > cap:
        idxs = np.linspace(lo, max_k, cap).astype(int)
    for k in idxs:
        c = s[k:k + CONTEXT]
        t = s[k + CONTEXT:k + CONTEXT + HORIZON]
        if len(c) == CONTEXT and len(t) == HORIZON:
            out.append((c.astype(np.float32), t.astype(np.float32)))
    return out


def make_windows_trainlast(s, stride, cap,
                           train_len=TRAIN_LEN, test_len=TEST_LEN):
    n = len(s)
    train_lo = max(0, n - train_len)
    test_lo = max(0, train_lo - test_len)
    train = _windows_in_block(s, train_lo, n, stride, cap)          # last 10y
    test = _windows_in_block(s, test_lo, train_lo, stride, cap)     # preceding 10y
    return train, test, test_lo, train_lo


def make_windows_split(s, stride, cap, cutoff_idx):
    """50/50: train windows target fully pre-cutoff; test windows context >= cutoff."""
    n = len(s) - (CONTEXT + HORIZON) + 1
    if n <= 0:
        return [], []
    idxs = np.arange(0, n, stride)
    if len(idxs) > cap:
        idxs = np.linspace(0, n - 1, cap).astype(int)
    train, test = [], []
    for k in idxs:
        c = s[k:k + CONTEXT]
        t = s[k + CONTEXT:k + CONTEXT + HORIZON]
        if len(c) != CONTEXT or len(t) != HORIZON:
            continue
        target_end = k + CONTEXT + HORIZON - 1
        context_end = k + CONTEXT - 1
        if target_end < cutoff_idx:
            train.append((c.astype(np.float32), t.astype(np.float32)))
        elif context_end >= cutoff_idx:
            test.append((c.astype(np.float32), t.astype(np.float32)))
    return train, test, cutoff_idx, cutoff_idx


def persistence_on_test(test_wins):
    errs, dirs, n = [], [], 0
    for cw, tw in test_wins:
        cl = cw[-1]
        P = np.full(HORIZON, cl)
        A = tw
        errs.append(np.abs(P - A) / np.abs(A).clip(min=1e-6))
        dirs.append(np.sign(A.mean() - cl) == np.sign(P.mean() - cl))
        n += 1
    if n == 0:
        return None
    return dict(mape=round(float(np.concatenate(errs).mean()) * 100, 2),
                dir_acc=round(float(np.mean(dirs)) * 100, 1), n=n)


def train_score_oos(train_wins, test_wins, steps, tag, pretrained=False):
    if len(train_wins) < 3 or len(test_wins) < 3:
        return dict(skipped=True, n_train=len(train_wins), n_test=len(test_wins))
    ctx = np.stack([w[0] for w in train_wins])[:, :, None]
    tgt = np.stack([w[1] for w in train_wins])[:, :, None]
    dl = DataLoader(TensorDataset(torch.tensor(ctx), torch.tensor(tgt)),
                    batch_size=BATCH, shuffle=True, pin_memory=True, drop_last=False)
    tctx = np.stack([w[0] for w in test_wins])[:, :, None]
    ttgt = np.stack([w[1] for w in test_wins])[:, :, None]
    dl_test = DataLoader(TensorDataset(torch.tensor(tctx), torch.tensor(ttgt)),
                         batch_size=BATCH, shuffle=False, pin_memory=True, drop_last=False)
    m = copy.deepcopy(BASE_MODEL)
    if pretrained:
        m.load_state_dict(pass4.warm, strict=False)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=gd.LR)
    s = 0; t0 = time.time()
    while s < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            o = m(past_values=xb, future_values=yb); loss = o.loss
            if not torch.isfinite(loss):
                print(f"    [NaN/inf loss in {tag}, aborting]"); s = steps; break
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); s += 1
            if s >= steps: break
        if s >= steps: break
    dt = time.time() - t0
    m.eval()
    p_all, a_all, cl = [], [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            out = m(past_values=xb.to(device))
            p = getattr(out, "prediction_outputs", out)
            if not isinstance(p, torch.Tensor):
                p = p[0] if isinstance(p, (tuple, list)) else out
            p_all.append(p.cpu().float().numpy()); a_all.append(yb.numpy())
            cl.append(xb[:, -1, 0].cpu().numpy())
    P = np.concatenate(p_all, 0).squeeze(-1); A = np.concatenate(a_all, 0).squeeze(-1)
    CL = np.concatenate(cl, 0)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mape = float((np.abs(P - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    mape_p = float((np.abs(np.repeat(CL[:, None], HORIZON, 1) - A) / np.abs(A).clip(min=1e-6)).mean() * 100)
    dir_acc = float((np.sign(A.mean(1) - CL) == np.sign(P.mean(1) - CL)).mean() * 100)
    mae = float(np.abs(P - A).mean())
    return dict(mape=round(mape, 2), mape_pers=round(mape_p, 2), dir_acc=round(dir_acc, 1),
                mae=round(mae, 3), n_train=len(train_wins), n_test=len(test_wins),
                secs=round(dt, 1), tag=tag)


P2_WIN = {
    "fixed200": dict(stride=1, cap=200),
    "scaled400": dict(stride=1, cap=400),
    "half_wstride256": dict(stride=256, cap=10_000_000),
    "quarter_wstride128": dict(stride=128, cap=10_000_000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=["AEP", "NVR", "FICO"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--mode", default="trainlast",
                    choices=["trainlast", "half"],
                    help="trainlast: train last 10y, test preceding 10y (production regime). "
                         "half: train first half, test second half.")
    ap.add_argument("--cutoff-frac", type=float, default=0.5,
                    help="for --mode half: fraction of history used for training")
    ap.add_argument("--strides", nargs="*", default=list(P2_WIN.keys()))
    args = ap.parse_args()

    results = []
    for tk in args.tickers:
        s = clean_series(tk, use_adj=True)
        n_total = len(s)
        print(f"\n=== {tk} (adjusted) n={n_total} mode={args.mode} ===", flush=True)
        pers = None
        for wname in args.strides:
            wp = P2_WIN[wname]
            if args.mode == "trainlast":
                tr, te, test_lo, train_lo = make_windows_trainlast(s, wp["stride"], wp["cap"])
            else:
                cutoff = int(n_total * args.cutoff_frac)
                tr, te, test_lo, train_lo = make_windows_split(s, wp["stride"], wp["cap"], cutoff)
            if not pers and te:
                pers = persistence_on_test(te)
                print(f"  [persistence baseline on test] {pers}", flush=True)
            if len(te) < 3:
                print(f"  [skip {wname}] only {len(te)} test windows", flush=True)
                continue
            tag = f"{tk}|{wname}"
            r = train_score_oos(tr, te, args.steps, tag, pretrained=False)
            r.update(part="P5-OOS", mode=args.mode, ticker=tk, win=wname,
                     steps=args.steps, use_adj="adj")
            results.append(r)
            print(f"  {tag:22} n_tr={r.get('n_train','-'):5} n_te={r.get('n_test','-'):5} "
                  f"MAPE={r.get('mape','-')}% (pers={pers['mape'] if pers else '-'}%) "
                  f"dir={r.get('dir_acc','-')}% MAE={r.get('mae','-')} {r.get('secs','-')}s",
                  flush=True)

    json.dump(results, open("/tmp/pass5_results.json", "w"), indent=2)
    print("\n=== PASS-5 DONE (out-of-sample) ===", flush=True)


if __name__ == "__main__":
    main()

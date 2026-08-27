#!/usr/bin/env python3
"""Device-throughput benchmark: CPU vs MX550 for tiny-model (TTM-class, ~1M param)
rolling-window time-series training. Isolates whether the MX550 gives a speedup
for our small-batch backfill workload. Uses a generic small model so it doesn't
depend on tsfm_public / the Granite checkpoint.
"""
import os, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PRICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_prices/")
CONTEXT, HORIZON = 512, 96
SAMPLE = ["AAPL","MSFT","NVDA","AMZN","GOOGL","TSLA","META","JPM","XOM","CVX",
          "JNJ","PG","HD","BAC","KO","PEP","WMT","DIS","V","MA"]

class TinyTSModel(nn.Module):
    """Approximates TTM-r2 scale: a small TCN-ish stack. ~1M params."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * CONTEXT, HORIZON)
    def forward(self, x):  # x: (B, CONTEXT, 1)
        x = x.permute(0, 2, 1)        # (B,1,CONTEXT)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x).unsqueeze(-1)  # (B,HORIZON,1)

def build_windows(df, tickers):
    wins = []
    for tk in tickers:
        s = df[df.ticker == tk].sort_values("date")["close"].values.astype(np.float32)
        if len(s) < CONTEXT + HORIZON:
            continue
        for k in range(len(s) - (CONTEXT + HORIZON) + 1):
            c = s[k:k+CONTEXT]; t = s[k+CONTEXT:k+CONTEXT+HORIZON]
            wins.append((c, t))
    return wins

def main():
    if os.environ.get("FORCE_CPU"):
        torch.cuda.is_available = lambda: False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = "cpu"
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = "cuda(unknown)"
    print(f"=== device: {device} ===  ({dev_name})")
    df = pd.read_parquet(PRICES)
    wins = build_windows(df, SAMPLE)
    print(f"windows: {len(wins)}")
    ctx = np.stack([w[0] for w in wins])[:, :, None]
    tgt = np.stack([w[1] for w in wins])[:, :, None]
    ds = torch.utils.data.TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model = TinyTSModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = 50
    t0 = time.time()
    model.train()
    for step in range(steps):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            break  # one batch per step
    dt = time.time() - t0
    print(f"{steps} steps (1 batch each, {len(dl.dataset)} wins): {dt:.1f}s on {device.type}")
    if device.type == "cuda":
        print(f"  peak GPU mem MiB: {torch.cuda.max_memory_allocated()//1024//1024}")

if __name__ == "__main__":
    main()

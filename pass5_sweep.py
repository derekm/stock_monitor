#!/usr/bin/env python3
"""
pass5_sweep.py — systematic OOS sweep for Granite-TTM.
Varies: train window length, strides, caps, steps, tickers, modes.
Outputs JSONL for analysis.
"""

import argparse, json, time, copy, subprocess, sys
import numpy as np, torch, pandas as pd

# Import from pass5 environment
import pass5

def run_experiment(ticker, train_window_years, test_window_years, stride, cap, steps, mode, cutoff_frac=0.5, use_pretrained=False, tag=None):
    """Run a single experiment configuration."""
    s = pass5.clean_series(ticker, use_adj=True)
    n_total = len(s)
    
    if mode == "trainlast":
        tr, te, _, _ = pass5.make_windows_trainlast(
            s, stride, cap, 
            train_len=train_window_years * 252,
            test_len=test_window_years * 252
        )
    else:
        cutoff = int(n_total * cutoff_frac)
        tr, te, _, _ = pass5.make_windows_split(s, stride, cap, cutoff)
    
    if len(te) < 3:
        return None
    
    # Persistence baseline
    pers = pass5.persistence_on_test(te)
    
    # Train & score
    exp_tag = tag or f"{ticker}|{mode}|{train_window_years}y|stride={stride}|cap={cap}|steps={steps}"
    r = pass5.train_score_oos(tr, te, steps, exp_tag, pretrained=use_pretrained)
    r.update({
        "ticker": ticker,
        "mode": mode,
        "train_window_years": train_window_years,
        "test_window_years": test_window_years,
        "stride": stride,
        "cap": cap,
        "steps": steps,
        "use_pretrained": use_pretrained,
        "pers_mape": pers["mape"] if pers else None,
        "pers_dir": pers["dir_acc"] if pers else None,
        "n_train": len(tr),
        "n_test": len(te)
    })
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/tmp/pass5_sweep.jsonl")
    ap.add_argument("--quick", action="store_true", help="Run quick subset for testing")
    ap.add_argument("--resume", action="store_true", help="Skip configs already in the output file")
    ap.add_argument("--max-experiments", type=int, default=None, help="Stop after N experiments (bounded runs)")
    args = ap.parse_args()

    # Define sweep space
    if args.quick:
        sweep = {
            "tickers": ["AEP", "NVR", "FICO"],
            "train_windows": [5, 10, 15, 20],
            "test_windows": [5, 10],
            "strides": [1, 128, 256],
            "caps": [100, 200, 400],
            "steps": [1500, 3000, 6000],
            "modes": ["trainlast"],
            "pretrained": [False]
        }
    else:
        sweep = {
            "tickers": ["AEP", "NVR", "FICO", "KO", "XOM", "JPM", "JNJ"],
            "train_windows": [3, 5, 7, 10, 15, 20],
            "test_windows": [3, 5, 10, 15],
            "strides": [1, 64, 128, 256],
            "caps": [50, 100, 200, 400, 800],
            "steps": [1500, 3000, 6000, 9000],
            "modes": ["trainlast", "half"],
            "pretrained": [False, True]
        }

    total = (len(sweep["tickers"]) * len(sweep["train_windows"]) * 
             len(sweep["test_windows"]) * len(sweep["strides"]) * 
             len(sweep["caps"]) * len(sweep["steps"]) * 
             len(sweep["modes"]) * len(sweep["pretrained"]))
    print(f"Total experiments: {total}", flush=True)
    if args.quick:
        print("Running QUICK sweep...", flush=True)

    done = 0
    seen = set()
    if args.resume:
        from pathlib import Path
        p = Path(args.output)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    key = (r.get("ticker"), r.get("mode"), r.get("train_window_years"),
                           r.get("stride"), r.get("cap"), r.get("steps"), r.get("use_pretrained"))
                    seen.add(key)
                except Exception:
                    pass
            print(f"Resume: {len(seen)} configs already done, skipping", flush=True)

    with open(args.output, "a" if args.resume else "w") as f:
        for tk in sweep["tickers"]:
            for tw in sweep["train_windows"]:
                for tew in sweep["test_windows"]:
                    for stride in sweep["strides"]:
                        for cap in sweep["caps"]:
                            for steps in sweep["steps"]:
                                for mode in sweep["modes"]:
                                    for pretrained in sweep["pretrained"]:
                                        # Skip invalid combos
                                        if mode == "half" and tw + tew > 30:
                                            continue
                                        if pretrained and mode != "trainlast":
                                            continue  # pretrained only makes sense in trainlast
                                        if args.resume and (tk, mode, tw, stride, cap, steps, pretrained) in seen:
                                            continue
                                        if args.max_experiments and done >= args.max_experiments:
                                            print(f"\nReached --max-experiments {args.max_experiments}; stopping", flush=True)
                                            f.flush()
                                            return
                                        done += 1
                                        if done % 50 == 0:
                                            print(f"Progress: {done}/{total}", flush=True)
                                        
                                        try:
                                            r = run_experiment(tk, tw, tew, stride, cap, steps, mode, 0.5, pretrained)
                                            if r:
                                                f.write(json.dumps(r) + "\n")
                                                f.flush()
                                                print(f"  {r['ticker']}|{r['mode']}|tr={r['train_window_years']}y|stride={r['stride']}|cap={r['cap']}|steps={r['steps']}|pre={r['use_pretrained']} -> MAPE={r.get('mape','-')}% dir={r.get('dir_acc','-')}% (pers={r.get('pers_mape','-')}%)", flush=True)
                                        except Exception as e:
                                            print(f"  ERROR {tk}|{mode}|tr={tw}|stride={stride}: {e}", flush=True)

    print(f"\n=== DONE: {done} experiments written to {args.output} ===", flush=True)

if __name__ == "__main__":
    main()
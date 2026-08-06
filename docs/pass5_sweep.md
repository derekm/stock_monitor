# pass5_sweep.py — resumable sweep

Systematic OOS sweep for Granite-TTM, now resumable and bounded.

## What changed (this round)

The full sweep space is 648+ experiments × ~3 min each ≈ 32h — larger than the
hardware realistically supports. Additions to make it tractable:

- `--resume` — skips configs already present in the output JSONL (keyed on
  ticker/mode/train-window/stride/cap/steps/pretrained). Crash-safe: a
  killed run can be continued without redoing completed experiments.
- `--max-experiments N` — bounded runs (stop after N new experiments).
- Output opens in append mode under `--resume` (previously truncated).

Note: the 2GB MX550 cannot fit parallel training processes; parallelizing the
sweep across processes would OOM. The resumable design is the honest speedup
here — run it in chunks between other work.

## Usage

```bash
python pass5_sweep.py --quick --resume --max-experiments 50
python pass5_sweep.py --resume
```

## Outputs

- JSONL of per-experiment results (see script header) — `--output`, default
  `/tmp/pass5_sweep.jsonl` (or `pass5_tier1.jsonl` for the quick tier).

## Related programs

- `pass5.py` — the honest-OOS harness each experiment runs
- `regime_forecast.py` — the regime-conditioned complement

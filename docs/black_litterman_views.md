# black_litterman_views.py

Builds Black-Litterman view vectors from the dual-pass / regime posture, then
runs the BL posterior to produce weights.

## Why it exists (rationale)

`black_litterman.py` takes manual `--view` arguments. This script automates
view construction: INCLUDE_CORE names get a bullish excess-return view, the
high-vol/stress regime shrinks views toward zero (less conviction), and
levered-asset names get their view dampened. It turns screen + regime state into
a ready-to-use view set, closing the loop from `preferred_metrics` and the HMM
regimes into `black_litterman.py`.

## Usage

```bash
python black_litterman_views.py --save
```

Flags: `--save` (write outputs). Reads `preferred_metrics.csv` and
`hmm_regimes.csv`.

## Outputs

- `black_litterman_views.csv` — auto-constructed views (ticker, view return,
  conviction)
- `black_litterman_weights_from_views.csv` — BL posterior weights from those views

(Schema families: screen_decision / weights_performance — see
[SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [black_litterman.md](black_litterman.md) — consumes the views
- [preferred_metrics.md](preferred_metrics.md) — INCLUDE_CORE source
- [hmm_regime_detection.md](hmm_regime_detection.md) — regime state input
- [inclusion_criteria.md](inclusion_criteria.md)

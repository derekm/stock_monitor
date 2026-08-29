# options_skew.py

Implied-vol skew and put/call volume ratios from the yfinance options chain.

## Why it exists (rationale)

Closes the "options IV skew, put/call ratios" TODO. `earnings_catalyst.py`
already fetches ATM IV; this adds the cross-sectional skew (vol smile tilt)
and put/call volume ratio — the classic fear/positioning indicators.

## Method

Per ticker, nearest-dated chain ~30d out:
- `atm_iv` — median IV of strikes within 5% of spot (sane IVs > 0.05;
  degenerate 1e-5 IV rows excluded)
- `skew` — IV(0.9×spot) − IV(1.1×spot); positive = downside puts richer
  (fear premium)
- `put_call_vol` — total put volume / total call volume (nearest expiry)

## Usage

```bash
python options_skew.py --save --max-tickers 60
```

## Outputs

- `options_skew.parquet` — per-ticker snapshot (date, spot, atm_iv, skew,
  put_call_vol, expiry).

## Related programs

- `earnings_catalyst.py` — the ATM-IV feed this extends
- `technical_signals.py` — sibling signal snapshotter

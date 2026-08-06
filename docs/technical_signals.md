# technical_signals.py

RSI, MACD, Bollinger/Keltner bands, SMA crossovers, and volume-price
confirmation across the monitored universe.

## Why it exists (rationale)

Closes the "technical signals" TODO. The repo had SMA alert rules but no
systematic indicator engine. Computed on split/dividend-adjusted closes
(`adj_close`) so indicators are comparable across splits.

## Indicators

- `rsi14` — Wilder RSI(14)
- `macd` / `macd_signal` / `macd_hist` — MACD(12,26,9)
- `bb_upper` / `bb_lower` / `bb_pct` — Bollinger(20, 2σ) + %B position
- `sma20` / `sma50` + `sma_cross` (+1 golden cross, −1 death cross)
- `above_sma20` / `above_sma50` — trend side
- Keltner/ATR computed internally (no TA-Lib dependency)

## Usage

```bash
python technical_signals.py --save                 # full universe
python technical_signals.py --save --tickers AAPL,MSFT
```

## Outputs

- `technical_signals.csv` — latest snapshot per ticker.

## Related programs

- `buy_candidates.py` — natural consumer (overlay)
- `momentum_analytics.py` — the momentum family this complements

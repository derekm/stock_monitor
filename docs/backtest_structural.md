# backtest_structural.py — backtest of structural / risk-scaled gate paradigms

## Why it exists (rationale)

The momentum-threshold ride gate (`ride_gate`) is a **lagging level detector**:
it opens after a surge, buys the top, then holds through the pullback. On
volatile / young / whipsaw names it loses to buy-and-hold. This script backtests
four structurally different entry paradigms plus two hybrids against the classic
momentum gate, all at **daily** frequency with **no lookahead** (position decided
at prior close, applied next day), using the **same `structural_positions()`
engine** that feeds the live `structural_gate` in `shock_ride.py`.

It answers: does a structural/risk-scaled gate beat the momentum gate and/or
buy-and-hold on raw return and drawdown?

## Paradigms (see `ride_longevity.structural_positions`)

- **turtle** — Donchian 55 breakout + 2×ATR chandelier stop
- **volscale** — target-vol-scaled exposure (0.30) gated by SMA200
- **regime** — EMA50/EMA200 markup/distribution state machine
- **recouple** — re-couple above EMA21 & EMA50, size by 1/vol
- **momentum** — classic daily momentum gate (baseline)
- **hybrid** — momentum entry + vol-scaled size + ATR stop
- **consensus** — majority of the four structural signals, vol-scaled

## Results (250 tickers, daily, no lookahead) — see docs/ride_longevity.md

- No timing approach beats buy-and-hold on raw return (BH 1403.8 vs best volscale 727.6).
- The gate's value is **drawdown control**: hybrid cuts mean maxDD to −0.433 vs
  −0.748 buy-hold; volscale has the best risk-adjusted (Calmar).
- On young/whipsaw names (RAL) the structural modes beat the momentum gate but
  still trail buy-and-hold.

## Usage

```bash
python backtest_structural.py --n 250
```

## Outputs

- `backtest_structural.parquet` — per-ticker per-paradigm `ride_return`,
  `buy_hold`, `excess`, `max_dd_ride`, `max_dd_bh`, `in_market`

## Related

- `ride_longevity.py` — `structural_gate` / `structural_positions` (the engine)
- `shock_ride.py` — live per-ticker ride screen (emits `structural_*` columns)
- `backtest_rides.py` — the monthly momentum-gate A/B (classic/quality/dual)

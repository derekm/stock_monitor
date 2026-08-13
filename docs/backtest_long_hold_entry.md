# backtest_long_hold_entry.py — ignition as a long-hold *entry*, not a timer

## Why it exists

A scratch on 100 tickers printed `gap_vol` entry-only at **+11.8% vs BH**.
That number is a **late-start artifact**: buy-and-hold is measured from the
first price, the strategy sits in cash ~40 days then holds forever, so its
compound wealth is not comparable. This script measures the same idea
honestly on the full universe.

## What buy-and-hold actually is

Hold from the first available date. No entry rule. No exit rule.

If you enter on a signal and **never exit**, return *from the entry date*
**equals** BH from that date (`excess_fair = 0` by construction). Timing
cannot improve a hold-forever book except by **skipping a worse early
window** (cash drag vs crash avoidance). That is the EW-portfolio test.

The useful market-wide question is therefore **cross-sectional**: when a
name ignites, is the next year better than the equal-weight universe?

## Variants

- `gap_vol` — gap up AND 20d volume z > 0.5
- `gap_vol_trend` — plus close > SMA200
- `gap_vol_regime` — plus SMA200 cross in last 20d
- `gap_vol_fresh` — no prior `gap_vol` in 63d (re-arm)
- `vol_persist` — volume z > 0.5 two days and 5d mom > 0
- `first_ignition` — first `gap_vol` in the recorded tape (IPO / listing bias)

T+1 fill. Entry-only (no ATR exit). Full monitored universe (583 names).

## Measured (as-of 2026-08-12)

Equal-weight daily book vs equal-weight universe: **+5 to +11 bp CAGR**.
Economically nothing. Drawdowns match BH (~−52%).

Event study (T+1 forward log excess vs EW universe):

| rule | 21d n | 21d ann xs | 252d n | 252d ann xs | 252d hit |
|------|------:|-----------:|-------:|------------:|---------:|
| gap_vol | 510,612 | +0.6% | 489,682 | +0.05% | 0.50 |
| gap_vol_trend | 336,610 | +0.3% | 323,054 | +0.57% | 0.51 |
| gap_vol_fresh | 3,127 | +2.4% | 3,112 | +0.15% | 0.51 |
| first_ignition | 578 | +13.3% | 576 | +6.9% | 0.58 |

`first_ignition` looks real and is **not** a long-hold entry rule to ship:
it is the first gap+volume print in the file, usually listing / IPO /
coverage start. Do not treat +6.9% as a repeatable add-on signal.

Repeated `gap_vol` on mature names is noise (half-million events, coin-flip).

## How to use it for long holds

Do **not** replace BH with “wait for gap_vol then hold forever.”

Do use ignition as a **fresh-capital / add-on screen** on top of the ride
stack that already exists:

true long-ride = BROAD + stack 4/4 + BUY/WATCH + no exit

plus a recent `gap_vol` (ideally trend-confirmed: close > SMA200).

Live quality names (ignition in the last ~10 sessions, as-of 2026-08-12):
HPE, DDOG, NUE, STLD, CRWD, FTNT (BUY + BROAD + 4/4). WATCH: TPR, GOLD, TGT.

## Usage

```bash
python backtest_long_hold_entry.py
python backtest_long_hold_entry.py --n 80 --fresh-days 10
```

Outputs: `backtest_long_hold_entry.parquet`, `long_hold_entry_screen.parquet`.
Research-only — not a daily JOB.

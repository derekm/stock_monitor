# shock_ride.py

Ride commodity/sector price explosions, exit before crisis — measured on
our own data.

## Why it exists (rationale)

The macro shock layers LABEL price explosions (oil +184% 1973-74,
fertilizer +232% 2007, coal +315% 2021). This script answers the strategy
question: can we FIND and RIDE these explosions, then GET OUT before the
crisis? The answer is measured, not asserted.

## Rule (per sector, monthly, no lookahead)

- **ENTER** when 12m basket momentum > entry threshold (default 0.40, the
  `elevated` band of macro_sector_shock) AND 3m momentum > 0
- **EXIT** when 3m momentum ≤ 0 (the rollover that precedes the collapse)
- position shifts 1 month after signals

Deliberately simple — entry = the shock layer's elevated band, exit =
trend rollover. No optimization.

## Measured results (full history, 14 sectors)

- **farming_inputs (fertilizer basket): ride +230.6% vs buy-hold +115.9%
  (+114.7 excess, 19 trades, in-market only 11.3% of the time)** — the
  canonical ride: the sector genuinely explodes and the rule catches it.
- **Overall: ride beats buy-hold in 1/14 on raw return** — most sectors'
  explosions are diluted by secular drift; the rule sits in cash through
  multi-decade uptrends.
- **The consistent win is drawdown: mean maxDD ride −30% vs buy-hold
  −68%** — the rollover exit gets out before the crises (e.g. flat through
  the 2008 fertilizer collapse that buy-hold ate whole).

## The honest strategy answer

The framework identifies explosions (the shock layers) and the rollover
exit avoids the crisis (halved drawdown). But riding is a **defensive
overlay on concentrated sector exposure, not a market-timing strategy**:
it wins big where the sector genuinely explodes (fertilizer) and otherwise
trades the secular drift for drawdown protection. Use it as: shock layer
labels the explosion → shock_ride times the exit → the barbell / defensive
book carries the sector exposure.

## Outputs

`shock_ride.csv` — per sector: `sector, n_trades, in_market_share,
buy_hold_return, ride_return, excess, max_dd_ride, max_dd_buyhold`

Reads: daily_prices.parquet, sp500_constituents.parquet (via
macro_sector_shock.SECTORS).

## Usage

```bash
python shock_ride.py --save [--entry 0.40]
```

Wired into `run_daily_automation.py` as the `taleb_shock_ride` job
(depends on `taleb_sector_shock`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

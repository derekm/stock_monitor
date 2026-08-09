# shock_ride.py — ride basket explosions, exit before crisis (DYNAMIC baskets)

## What it does

For **every dynamic basket** (GICS sector / sub-industry / factor group —
see macro_sector_shock.py), measures a simple ride rule against buy-hold:

- ENTER when 12m basket momentum > 0.40 AND 3m momentum > 0
- EXIT when 3m momentum ≤ 0
- position shifts 1 month after signals (no lookahead)

## Honest measured results (full history, 161 dynamic baskets)

- **2/161 baskets beat buy-hold on raw return** (fg_industry_solar +105%,
  fg_industry_telecom_services +68%) — riding is a defensive overlay on
  concentrated exposure, not a market-timing strategy.
- **Drawdown is the consistent win: mean maxDD ride −40% vs buy-hold
  −78%.** The rollover exit gets out before the crises.
- The rule wins big where a basket genuinely explodes (solar) or collapses
  (telecom services) and otherwise trades secular drift for drawdown
  protection.

## Outputs

`shock_ride.csv` — per basket: `basket, basket_kind, label, n_members,
n_trades, in_market_share, buy_hold_return, ride_return, excess,
max_dd_ride, max_dd_buyhold`

`shock_ride_tickers.csv` — **per-ticker** ride pass over the full price
universe (min 36mo history): `ticker, name, sector, n_trades,
in_market_share, buy_hold_return, ride_return, excess, max_dd_ride,
max_dd_buyhold, mom1, mom3, mom12, ride_long, recommendation,
interpretation, as_of`

## Usage

```bash
python shock_ride.py --save [--entry 0.40]
```

Wired into `run_daily_automation.py` as `taleb_shock_ride`; feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

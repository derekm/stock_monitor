# factor_rotation_defense.py

Defensive factor-rotation strategies across quality / value / low-vol /
momentum / dividend / dual-pass / annotation-group sleeves, rotated by a
risk-on/off signal + a value-momentum overlay.

## Why it exists (rationale)

Rather than a static book, rotate factor sleeves with the regime: overweight
quality/dual in risk-on, low-vol + dividend ETFs in risk-off (high vol or
crisis flag), and hold a momentum sleeve when 12-1 momentum is strong — the
Value-and-Momentum-Everywhere finding (Asness/Moskowitz/Pedersen 2013) that
value and momentum are strongly negatively correlated (~-0.55) and are the
diversification halves of the factor pair.

## The two-table group system (2026-08)

Named groups are NOT hardcoded ticker lists. They live in two CSV tables
seeded from the per-stock annotations + the S&P 500 change history; adding a
group = appending rows, no code change:

- `factor_groups.csv` — **catalog**: (group, group_type); types are
  `sector`, `industry`, `index`, `sleeve`, `dynamic`, `custom`. Seeded from
  monitored_stocks `sector`/`industry`/`value_sleeve`/`defensive_value_index`/
  `growth_tech_index`/`dual_pass_member` + GICS for SP500 names.
- `factor_group_members.csv` — **join with as-of dates**:
  (group, ticker, valid_from, valid_to). Membership is point-in-time: a
  ticker belongs on date d iff `valid_from <= d` and (`valid_to` null or
  `valid_to > d`). S&P 500 memberships are temporal windows built from the
  `sp500_changes.parquet` event timeline (a member iff the latest event at or
  before d is an ADD) — so the `sp500` sleeve's composition evolves with real
  additions/removals.

Every group in the members table becomes a sleeve automatically.

## The temporal model — removals in every time frame

`valid_to` is universal: ANY group can carry removal dates, not just sp500.
A membership is active on date d iff `valid_from <= d` and (`valid_to` null
or `valid_to > d`). Groups that don't need history just leave both dates
blank (always-valid) — evict by deleting the row or setting `valid_to`.

The sp500 group is fully temporal from `sp500_changes.parquet`: contiguous
windows walked from the sorted add/remove timeline (a ticker is a member on
d iff the latest event at or before d is an ADD). Verified with real
removals: GM [1957→2009] then [2013→open] (bankruptcy + return), DELL
[1996→2013] then [2024→open] (private + re-IPO), and the index size evolves
437 (2000) → 484 (2010) → 508 (2026).

## Group editor CLI

Groups grow without code changes; the CLI edits the two tables:

```bash
python factor_rotation_defense.py add-group --group my_basket --type custom
python factor_rotation_defense.py add --group my_basket --ticker AEP --from 2020-01-01
python factor_rotation_defense.py add --group my_basket --ticker AEP --to 2024-06-30
python factor_rotation_defense.py evict --group my_basket --ticker AEP            # delete rows
python factor_rotation_defense.py evict --group my_basket --ticker AEP --on 2024-06-30  # dated eviction
python factor_rotation_defense.py show --group sp500 --as-of 2010-06-30          # PIT view
python factor_rotation_defense.py timeline --ticker GM                           # all windows
```

The `run` subcommand (default: `python factor_rotation_defense.py run --save`)
executes the rotation with current group tables.

## Universe honesty (2026-08 audit)

- quality/value/dual are rebuilt **point-in-time** from `fundamentals.parquet`
  as-of each month-end (was: latest-fundamentals membership applied to all
  history — a look-ahead bias).
- low_vol + momentum are computed over ALL price tickers (not the
  fundamentals subset).
- Coverage: 142 monitored tickers carry full annotations, 503 SP500 carry
  GICS, 549 carry fundamentals, 551 carry prices.

## Usage

```bash
python factor_rotation_defense.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `monitored_stocks.parquet`,
`fundamentals.parquet`, `sp500_constituents.parquet`, `sp500_changes.parquet`;
writes `factor_groups.csv` + `factor_group_members.csv` on first run.

## Outputs

- `factor_rotation_weights.csv` — target sleeve weights
- `factor_rotation_performance.csv` — backtest performance per sleeve
- `factor_sleeve_returns.csv` — sleeve return series (PIT membership)
- `factor_groups.csv` / `factor_group_members.csv` — the group catalog + join
  table (seeded on first run; editable)

(Schema families: weights_performance / base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

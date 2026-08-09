# shock_ride.py — ride basket/ticker explosions, exit before crisis (DYNAMIC)

## What it does

For **every dynamic basket** (GICS sector / sub-industry / factor group —
see [macro_sector_shock.md](macro_sector_shock.md)) **and every ticker**
in the price universe (min 36mo history), measures the same ride rule against
buy-hold:

- **ENTER** when 12m momentum > 40% AND 3m momentum > 0
- **EXIT** when 3m momentum ≤ 0
- position shifts 1 month after signals (no lookahead)

The rule is deliberately simple: entry = the shock layer's `elevated` band,
exit = trend rollover. No optimization — the point is to show whether the
shock framework has exploitable timing, with honest numbers per basket/ticker.

## Formulas

**Monthly returns:**

$$
r_\tau = \ln\left(\frac{C_\tau}{C_{\tau-1}}\right)
\quad\text{where}\quad
C_\tau = \prod_{s=1}^\tau \left(1 + \bar{r}_s\right)
\quad\text{and}\quad
\bar{r}_s = \frac{1}{|B_s|} \sum_{i \in B_s} \ln\left(\frac{P_{i,s}}{P_{i,s-1}}\right)
$$

**Momenta:**

$$
\text{mom}_{12}(\tau) = \frac{C_\tau}{C_{\tau-12}} - 1
\qquad
\text{mom}_3(\tau) = \frac{C_\tau}{C_{\tau-3}} - 1
\qquad
\text{mom}_1(\tau) = \frac{C_\tau}{C_{\tau-1}} - 1
$$

**Ride rule (per basket / per ticker, monthly, no lookahead):**

$$
\text{long}(\tau) = \mathbb{1}\left[\text{mom}_{12}(\tau-1) > 0.40
\quad\land\quad
\text{mom}_3(\tau-1) > 0\right]
$$

Position enters the month *after* signals — no lookahead.

**Exit:** when $\text{mom}_3(\tau) \le 0$ (rollover).

**Ride return vs buy-hold:**

$$
\text{ride} = \sum_\tau \text{long}(\tau) \cdot r_\tau
\qquad
\text{BH} = \sum_\tau r_\tau
\qquad
\text{excess} = \text{ride} - \text{BH}
$$

**Max drawdown:**

$$
\text{maxDD} = \min_\tau \left(\frac{C_\tau}{\max_{s \le \tau} C_s} - 1\right)
$$

**Current-state recommendation (honest — same logic as [ride_now.md](ride_now.md)):**

| Recommendation | Condition |
|---|---|
| **BUY** | ride_long ∧ mom12 > 0.40 ∧ mom3 > 0 ∧ mom1 > 0 |
| **STAND DOWN** | ride_long ∧ (mom3 ≤ 0 ∨ mom1 ≤ 0) — momentum says long but 1m rolling over |
| **AVOID** | mom12 > 0.40 ∧ mom3 ≤ 0 — exploded but rolled over (ride exited) |
| **WATCH** | mom12 > 0.40 ∧ mom3 > 0 ∧ ride_long = 0 — above threshold but 3m not yet positive |
| **FLAT** | otherwise |

## Honest measured results (full history, 161 dynamic baskets + 572 tickers)

| Metric | Baskets | Tickers |
|---|---|---|
| Beat buy-hold on return | 2 / 161 | 34 / 572 |
| Mean excess vs BH | −434% | −280% |
| Mean maxDD ride vs BH | **−40% vs −78%** | **−41% vs −81%** |

**The consistent win is drawdown protection, not return enhancement.**
The rollover exit gets out before the crises (subindustry_regime: stress
coincident with rollover ~14-17%, not leading).

## Outputs

- `shock_ride.csv` — per basket: `basket, basket_kind, label, n_members,
  n_trades, in_market_share, buy_hold_return, ride_return, excess,
  max_dd_ride, max_dd_buyhold`
- `shock_ride_tickers.csv` — **per-ticker** ride pass over the full price
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

- [macro_sector_shock.md](macro_sector_shock.md) — dynamic baskets source
- [subindustry_regime.md](subindustry_regime.md) — per-basket HMM stress (stress gate)
- [macro_sector_shock.md](macro_sector_shock.md) — shock labels (entry gate)
- [ride_now.md](ride_now.md) — current-state recommendations
- [hmm_regime_detection.md](hmm_regime_detection.md) — stress posterior source
- [macro_fragility.md](macro_fragility.md) — macro fragility context
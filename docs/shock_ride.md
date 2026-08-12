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
- `shock_ride_tickers.parquet` — **per-ticker** ride pass over the full price
  universe (min 3mo history; classic rule for names with >=36mo, else the
  young-ticker gate): `ticker, name, sector, n_trades,
  in_market_share, buy_hold_return, ride_return, excess, max_dd_ride,
  max_dd_buyhold, mom1, mom3, mom12, ride_long, recommendation,
  interpretation, as_of`, plus the research-momentum columns:
  `is_young, tsmom_3mo_sharpe, tsmom_6mo_sharpe, tsmom_12mo_sharpe,
  stmom_1m_ret, gw_high_prox, young_gate_open, young_gate_reliability`,
  and the fresh-breakout columns: `fresh_verdict (FRESH_BREAKOUT / BUILDING /
  MATURING / EXHAUSTED / NO_SIGNAL), fresh_score, fractal_agreement`

The recommendation now layers the **fresh-breakout detector**
([`breakout_detector.md`](breakout_detector.md)) and **fractal consensus**
([`fractal_windows.md`](fractal_windows.md)) onto the classic ride rule:
a BUY requires the breakout to be FRESH (near-high + accelerating) where
available; EXHAUSTED breakouts (near-high + volume divergence) are AVOID.

## Young-ticker gate (research-grounded, <36mo history)

Newly listed / recently onboarded tickers lack the 12-month window the classic
rule needs. `shock_ride.py` now applies the **graduated young-ticker gate** from
[`momentum_research.py`](momentum_research.md) to names with <36 months:

- requires ≥6 months clean history (strict min 3; the first ~1 month is dropped
  per Ritter 1991 — IPO pop is not momentum)
- annualized 3/6-mo momentum vs a maturity-scaled 40% gate
- requires 6-mo momentum > 0 AND 1-mo return > 0 (RFS 2022 liquid-stock
  continuation) AND nearness to the listing all-time-high (George-Hwang 2004)
- volatility + liquidity filters (short-term momentum only works in liquid,
  high-turnover names)

For young names the recommendation uses the gate (BUY when open, FLAT with the
reasons when not) instead of the 12-month rule. The research measures
(TSMOM 3/6/12, JT-6, STMOM-1, GW-high) are emitted for every ticker regardless
of age.

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
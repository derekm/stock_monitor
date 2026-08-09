# ergodicity_ruin.py

Portfolio ergodicity & ruin analysis — the fat-tail complement to `tail_index.py`.
Averages over paths ≠ averages over time when tails are fat. This layer
quantifies the **ergodicity gap** and the **ruin probability** that
ensemble averages hide.

## Formulas

**Tail index (from tail_index.py):**

$$
\alpha \approx \frac{1}{\text{Hill}(r)}
$$

where $\alpha$ = tail exponent; $\alpha < 3$ ⇒ fat tails (finite variance,
infinite higher moments); $\alpha \to 2$ ⇒ Cauchy territory.

**Ruin probability at horizon $H$:**

$$
P_{\text{ruin}}(H) = P\left(\min_{1 \le h \le H} W_h < 0.5 W_0\right)
$$

Estimated via Monte Carlo using the fitted tail model (Pareto above $u$ +
Gaussian body). Equivalent: $P(\text{terminal wealth} < 0.5 W_0)$.

**Ergodicity gap (ensemble vs time average):**

For return $r$ with tail exponent $\alpha$:

$$
\frac{\mathbb{E}[r^2]}{(\mathbb{E}[r])^2} \propto \frac{1}{N^{2/\alpha - 1}}
$$

The gap explodes as $N \to \infty$ when $\alpha < 2$ — ensemble average
diverges while time average stays finite.

**Ruin metrics (per horizon $H$):**

| Metric | Formula |
|---|---|
| Ruin probability | $P(\min_{h \le H} W_h < 0.5 W_0)$ |
| P(99% drawdown) | $P(\min_{h \le H} W_h < 0.01 W_0)$ |
| Days to double | median $t: W_t \ge 2 W_0$ |
| Days to 5% drawdown | median $t: W_t < 0.95 W_0$ |
| Double/ruin ratio | median(double time) / median(ruin time) |

**Portfolio ergodic metrics (equal-weight portfolio):**

Same metrics computed on the equal-weight portfolio of monitored tickers.

## Outputs

- `tail_risk_hedge_crisis.csv` — per ticker: `ticker, n_obs, tail_alpha_hill,
  emp_p_gt_3sd, gauss_p_gt_3sd, tail_ratio_3sd, emp_p_gt_5sd, gauss_p_gt_5sd,
  tail_ratio_5sd, kurtosis`
- `portfolio_ergodic.csv` — equal-weight portfolio: same metrics + terminal
  wealth p5/p50/p95 per horizon

## Usage

```bash
python ergodicity_ruin.py --save
```

Registered as the `taleb_ergodic` daily job (after `tail`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [tail_index.md](tail_index.md) — tail alpha input
- [barbell_check.md](barbell_check.md) — convexity allocation driven by ergodicity gap
- [gap_risk.md](gap_risk.md) — gap share complements the tail analysis
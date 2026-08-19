# fragility_screen.py

Per-name fragility screening — the micro half of the Taleb layer (macro
half is `macro_fragility.py`). Every monitored ticker gets a composite
fragility score composed of independent drivers, each noise-robust.

## Formulas

**Driver z-scores (cross-sectional, per date):**

For each driver $d$ in {leverage, tail_alpha, gap_share,
illiquidity, iv_skew, kurtosis}:

$$
z_d(i,t) = \frac{x_d(i,t) - \mu_d(t)}{\sigma_d(t)}
$$

where $\mu_d(t), \sigma_d(t)$ are cross-sectional mean/std across the
monitored universe at date $t$.

**Composite fragility score (noise-robust):**

$$
fragility(i,t) = \sum_{d} contribution_d(i,t)
$$

where each driver's contribution is the **noise-convolved expectation** (from
[hidden_optionality_audit.md](hidden_optionality_audit.md)):

$$
contribution_d = \mathbb{E}_{z \sim \mathcal{N}(0, \sigma_d)}[f_d(x_d + z)]
$$

with $\sigma_d = \sigma_{x_d} / 4$ (cross-sectional std / 4).

**Fragile flag:**

$$
fragile_flag(i,t) = \mathbb{1}[fragility(i,t) > pctile_{95}(t)]
$$

Top 5% most fragile names flagged.

**Skew steepening penalty (for buy_candidates):**

Names with high IV skew get an additional `-0.15` to their composite score
in `buy_candidates.py` (the `skew_steepening` driver).

## Outputs

`fragility_screen.csv` — `ticker, date, fragility_score, fragility_pctile,
fragile_flag, leverage_pct, asset_coverage_pct, interest_coverage_pct,
iv_skew_pct, illiquidity_pct, gap_share_pct, tail_fatness_pct, kurtosis_pct`

## Usage

```bash
python fragility_screen.py --save
```

Registered as the `taleb_fragility` daily job (after `taleb_tail` +
`taleb_gap`; feeds `export` and `buy_candidates.py`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [tail_index.md](tail_index.md) — tail alpha input
- [gap_risk.md](gap_risk.md) — gap share input
- [options_skew.md](options_skew.md) — IV skew input
- [buy_candidates.md](buy_candidates.md) — consumes fragility + skew steepening
- [shadow_book.md](shadow_book.md) — fragile names trigger kill switch
- [barbell_check.md](barbell_check.md) — average fragility scales convexity allocation
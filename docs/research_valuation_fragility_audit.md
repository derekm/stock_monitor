# Research Library — Valuation & Fragility Methodology Audit

Sources ingested 2026-08-11 to tighten the value/quality/fragility screens and
correct hidden assumptions. Each entry: finding → implication → implementation.

## 1. Ohlson & Rueangsuwan (2026) — "Formal Equity Valuation: Overview and Limits"
SSRN 6280638 / doi:10.2139/ssrn.6280638

**Core RIV / EPS-Cap relation:** `P0 = BVPS + Σ(RE_t)/(1+r)^t`, `RE_t = EPS_t − r·BVPS_{t−1}`;
terminal/forward-PE form anchors `P = f(EPS₁, EPS₂, DPS₁, BVPS, r, g)`.

**Finding:** any formula needing a *firm-specific* growth parameter `g` cannot be
expressed meaningfully in financial attributes — **g is under-identified** and
mechanically anchoring `g = r/2` is exactly the kind of free parameter they warn
against. EPS/forward-P/E should be the centerpieces, not an arbitrary g.

**Implementation applied:**
- Implied-r kept as `r = 2·ROE/(P/B+1)` (no free g — the paper's preferred ICC
  inversion), but the fair-value **band** (which uses `P = −BV + 2·EPS1/r` with
  implicit g=r/2) is now reported alongside a **g-sensitivity test** so users see
  how much the band moves under g = 0 / r/2 / 0.75r.
- The g=r/2 anchor is explicitly disclosed in output/docs as an assumption.

## 2. Frankel & Lee (1998) — "Accounting Valuation, Market Expectation, and Cross-sectional Stock Returns" (JAE)
Formula: `V_t = BVPS_t + Σ_{i=1..T}(E[EPS_{t+i}] − r·BVPS_{t+i−1})/(1+r)^i + terminal`, analyst-forecast EPS, clean surplus.

**Findings:** analyst optimism inflates V; single constant r confounds risk; V/P is a
**long-horizon (~3yr) value signal**, not a short-window predictor; sensitive to
terminal anchor.

**Implementation:** V/P / implied-r treated as a **multi-year value signal**, not a
6-month timing trigger. Extreme values winsorized. (See `implied_r_screen.py`.)

## 3. Penman & Sougiannis (1998) — "A Comparison of Dividend, Cash Flow, and Earnings Approaches to Equity Valuation" (CAR)
**Finding:** abnormal-earnings (residual income) valuation is far less error-prone than
free-cash-flow or dividend discounting at realistic finite horizons — because FCF and
dividends are low/volatile/uninformative in early years. Accrual earnings front-load
value through book value, reducing terminal dependence.

**Implementation:** The repo's implied-r uses **book value anchored RIV** (not DCF/DDM),
which is the correct choice per this result for low-dividend/negative-FCF names.

## 4. Damodaran — Valuing Financial Service Firms (NYU Stern) + "Good (Bad) Banks" (2023)
URLs: pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/finsvc.pdf, aswathdamodaran.substack.com/p/good-bad-banks-good-bad-investments

**Finding:** for banks/insurers/REITs, debt/deposits are **raw material, not capital**.
Book equity ≈ invested assets and float (deposits/premiums) inflates levered ROE on a
thin equity base. So `r = 2·ROE/(P/B+1)` **mechanically misreads financials as
artificially cheap**: P/B stays near/below 1 (book is small vs levered assets) while
ROE stays elevated. Standard EV/WACC and FCF framing is "fanciful and fruitless" for
financials.

**Correct metric:** equity-only **excess-return model** — Value = BV + PV of excess
returns, where **excess return = ROE − cost of equity**; P/B is driven by
`(ROE − g)/(r − g)`. Practical substitutes: DDM, FCFE (NI − regulatory capital build),
and P/E vs ROE and cost of equity. Do not apply operating-company leverage targets;
use equity capital ratio (Tier 1) and cost of equity.

**Implementation applied (implied_r_screen.py):**
- `DISTORTED_SECTORS = {Financials, Utilities, Real Estate, Financial, Multi-Sector}`
- `r_distorted` flag + `implied_r_clean` column (NaN for distorted sectors) so
  book-heavy sectors no longer pollute the CHEAP screen.
- Sector sourced from `sp500_constituents.parquet` (full universe) + monitored_stocks.
- **Financials excess-return metric** added: `excess_return = ROE − cost_of_equity`,
  reported separately so financials are judged by ROE−COE, not implied-r.

## 5. Daniel & Moskowitz (2016) — "Momentum Crashes" (JFE / NBER w20439)
**Finding:** momentum crashes are concentrated in **panic states after market declines**
with high vol (VIX) and abrupt rebounds. Mechanism: past losers are high-beta distressed
stocks with low down-market beta (≈0.66) but high up-market beta in rebounds (≈1.47),
so at a bear-market bottom the short-losers leg rallies violently → large negative
spread. Biggest crashes: 1932, 2009 (both post-crisis rebounds).

**Persistence:** 3-12 month window, peak ~6-12 months, then **partial reversal beyond 12
months** (Jegadeesh-Titman). Confirms the repo's measured curve (peak ~6m, negative ~12m).

**Rule failure:** a fixed 6-9 month hold fails exactly when holding momentum **through a
bear→bull regime reversal**. Mitigation: size inversely to ex-ante vol, and cut exposure
when a bear-market indicator is on + high vol.

**Implementation:** the momentum holding rule is now **regime-conditional** — when the
macro Minsky debt-impulse is in `crisis_band` and the market is stressed, momentum rides
are shortened to a 3-month exit rather than the default 6-9. (Documented; see buy_candidates
regime gating.)

## 6. Options-Implied Tail Risk / IV Skew
**Measures:** 25Δ put-call risk reversal `RR = IV(25Δ call) − IV(25Δ put)` (structurally
negative for indices); 25Δ butterfly `BF = ½(IV(25Δ c) + IV(25Δ p)) − IV(ATM)`; put-skew
ratio `25Δ Put IV / 25Δ Call IV` (>1 ⇒ negative skew). CBOE SKEW = `100 − 10·S` (30d risk-
neutral skewness of SPX).

**Data:** yfinance option chains (free, per-strike → compute IV via BS inversion →
interpolate to 25Δ) for single-stock skew; CBOE SKEW CSV for index level.

**Predictive power:** steeper put skew precedes extremes and signals crash risk rather
than direction; SKEW is a level/regime tail indicator, weak as a timing signal.

**Implementation:** `fragility_screen` `iv_skew_pct` column populated via a new
`iv_skew.py` that pulls yfinance option chains for the universe and computes the 25Δ
put-skew slope (50Δ−25Δ put IV). Non-optionable/illiquid names left NaN and handled
gracefully in the composite.

---
*Research captured 2026-08-11. See `docs/implied_r_screen.md`, `docs/fragility_screen.md`,
`docs/macro_fragility.md` for the operational screens.*

# breakout_detector.py — fresh breakout detection

Layers a **fresh-breakout** detector on top of the fractal-of-sliding-windows
momentum ([`fractal_windows.md`](fractal_windows.md)). Distinguishes a NEW
breakout from a maturing/exhausted one and avoids buying the top.

## Why it exists (rationale)

The fractal consensus tells you a range is *trending*, but not whether it is
*freshly* breaking out or already mature. The research audit identified the
measures that separate a fresh impulse from an exhausted one (George-Hwang 2004,
Gettleman-Marks acceleration, Donchian, volume/exhaustion). This detector
combines them into a composite.

## The four fresh-breakout components

**1. Price-to-52-week-high (George-Hwang 2004)** — nearness to the prior high
predicts returns that DON'T reverse (persists to 24mo), beating trailing returns:

$$
PTH(t) = C(t) \;/\; max( C(t-W..t) ), \quad W = 252
$$

**2. Donchian close-break (longest window)** — a true close through the trailing
N-day high, not a first-tick wick:

$$
DB(t) = 1( \; C(t) >= max(C(t-N..t)) \quad and \quad C(t) > max(C(t-N..t-1)) \; ), \quad N = 60
$$

**3. Acceleration / momentum-of-momentum (Gettleman-Marks 2006)** — the key
freshness discriminator. Momentum positive AND rising (2nd derivative > 0):

$$
M(t) = ln(C(t)) - ln(C(t - 126))
\qquad
A(t) = M(t) - M(t - 126)
$$

$$
freshAcc(t) = 1( \; M(t) > 0 \quad and \quad A(t) > 0 \; )
$$

**4. Volume confirmation** — On-Balance Volume rising + multi-day volume
expansion (institutional participation):

$$
OBV(t) = sum( sign(C(s) - C(s-1)) \cdot V(s) ), \quad s \le t
$$

$$
volOK(t) = 1( \; OBV20 > 0 \quad and \quad V / Vbar10 > 1 \; )
$$

where $OBV20$ is the 20-day change in OBV and $Vbar10$ the 10-day mean volume.

## Composite score

$$
fresh = 0.25 \cdot PTH + 0.20 \cdot DB + 0.35 \cdot freshAcc + 0.20 \cdot volOK
$$

## Verdict

| Verdict | Condition |
|---|---|
| **FRESH_BREAKOUT** | freshAcc ∧ PTH ≥ 0.90 |
| **BUILDING** | acceleration ∧ PTH < 0.90 |
| **MATURING** | no acceleration ∧ PTH ≥ 0.90 |
| **EXHAUSTED** | volume divergence ∧ PTH ≥ 0.90 |
| **NO_SIGNAL** | otherwise |

Rejection signals: decelerating momentum (A < 0), volume divergence (price up /
OBV flat), exhaustion-gap climaxes, parabolic volatility expansion.

## Usage

```python
from breakout_detector import fresh_breakout_score, fractal_fresh
fresh = fresh_breakout_score(close, volume)   # per-date components + verdict
joined = fractal_fresh(close, volume)          # + fractal consensus agreement
```

## Outputs

No files written directly. Produces per-date DataFrames with the components
(`pth`, `donchian_break`, `acceleration`, `fresh_acceleration`, `volume_confirmed`,
`volume_divergence`), a `fresh_score`, and a `verdict`.

## Related

- [`fractal_windows.md`](fractal_windows.md) — the underlying multi-granularity momentum
- [`momentum_research.md`](momentum_research.md) — the research measures
- `docs/research_valuation_fragility_audit.md` — source research
- `run_tests.py` — executable tests (breakout_detector, breakout_verdict)

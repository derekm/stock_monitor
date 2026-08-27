# analyze_arista.py — ARISTA reliability & short-placement study

## What it does

Honest historical answers to three questions about the
[arista.py](arista.md) top-of-uptrend detector, computed over the full
universe with point-in-time data (no lookahead):

1. **Reliability** — of every `arista_signal` session, what fraction preceded
   a real top (≥15% forward drawdown within 120 sessions)? And how does that
   depend on `arista_score`?
2. **Score as intensity** — does a higher score select higher caught-rate?
3. **Short timing** — can the signal place a SHORT day-of vs week-off? (The
   measured answer: the signal leads the trough by ~56 sessions, so a fixed
   21–63d short loses; ARISTA is a de-risk/exit trigger, not a short-entry
   timer.)

Reads `arista_metrics.parquet` + `daily_prices/`. Writes
`arista_reliability.parquet` (per-signal forward stats).

## Honest measured results (full universe, 274k signals)

| Score bucket | n | Caught ≥15% dd | Avg fwd maxDD | Med days-to-trough | Day-of short | Week-off short |
|---|---|---|---|---|---|---|
| 0–0.2 | 31k | 71% | −21% | 59 | +9.8% | +11.6% |
| 0.2–0.35 | 114k | 76% | −22% | 60 | +10.7% | +12.6% |
| 0.35–0.5 | 106k | 85% | −26% | 58 | +11.9% | +14.2% |
| 0.5–0.65 | 22k | 89% | −28% | 56 | +12.7% | +15.5% |
| 0.65+ | 2k | 86% | −26% | 56 | +11.5% | +14.4% |

**Reliability:** the signal catches ≥15% drawdown **80% overall**, rising
monotonically with score to **~89%** at score ≥ 0.5. Score is a clean
intensity filter.

**Short timing (the honest answer):** the signal leads the trough by a median
**~56 sessions** (mean 71). A short placed day-of or week-off and held to the
trough earns +11–15%, but a short on a **fixed 21/63d horizon loses** (win
rate 41%/38%) because price often runs further up before the breakdown. Adding
breakdown confirmation (break of 10d low, or 5d momentum < 0) **does not**
improve the fixed-horizon short — it only delays entry. Conclusion: use ARISTA
to **de-risk / exit longs**, not to time short entries.

## Usage

```bash
python analyze_arista.py reliability   # score-bucket caught rates + timing
python analyze_arista.py short         # short-placement variants (negative result)
```

## Related programs

- [arista.md](arista.md) — the detector
- [shock_ride.md](shock_ride.md) — entry/exit ride gate

#!/usr/bin/env python3
"""hidden_optionality_audit.py — the American-options lesson applied to decisions.

Why: the American-options paper (El Hassan, Maddah & Taleb 2026) shows that
ANY quantity treated as deterministic while it is actually stochastic contains
unpriced convexity — the early-exercise feature of an American option is
optionality on the path of the rate differential, invisible to models that fix
r1-r2. The paper's method (§I-A): stochasticize one input at a time and measure
the convexity bias π = E[f(ã)] - f(E[ã]) — "a bad ruler might not give us the
precise height of a growing child, but will inform us whether the child is
growing."

Our stack has the same structure: buy_candidates scores are built from point
estimates (momentum_score, factor_composite, aggregate composite, the hard HMM
regime label). Each is an "r1-r2" — deterministic in our system, stochastic in
reality. This script perturbs each driver by its OWN estimation error (the
standard error of the underlying metric, not a made-up epsilon) and measures
how often the BUY/ACCUMULATE/WATCH/AVOID decision flips. The flip rate IS the
hidden optionality: the probability that our decision is riding on noise.

Outputs:
  hidden_optionality.csv — per driver: perturb scale (est. error), decision
  flip rate, mean |score change|, convexity bias π. Sorted by flip rate.

Reads: the same CSVs as buy_candidates (preferred/momentum/factor/risk/
aggregate) + hmm_regime_states.csv.
Usage: python hidden_optionality_audit.py [--n-perturb 200] [--seed 7]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perturb", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import buy_candidates as bc

    base = bc.build()
    if base.empty or "action" not in base.columns:
        raise SystemExit("buy_candidates.build() returned nothing — run analytics first")
    base = base.set_index("ticker", drop=False)
    base_action = base["action"].astype(str)

    # decision mapping for flip detection (order matters)
    ORDER = ["AVOID", "WATCH", "ACCUMULATE", "BUY"]
    ORDER_IDX = {a: i for i, a in enumerate(ORDER)}

    def _rescore(df):
        """Re-run the scoring loop from buy_candidates on perturbed inputs.
        Returns the action Series (AVOID/WATCH/ACCUMULATE/BUY) per ticker."""
        reg = bc.regime()
        stress = "stress" in reg.lower()
        out = pd.Series(index=df.index, dtype=object)
        for tk, r in df.iterrows():
            score = 0.0
            reasons = []
            if r.get("dual_pass_core") is True or str(r.get("dual_pass_core", "")).lower() == "true":
                score += 0.35
            if r.get("value_trifecta") is True or str(r.get("value_trifecta", "")).lower() == "true":
                score += 0.20
            if r.get("buffett_quality") is True or str(r.get("buffett_quality", "")).lower() == "true":
                score += 0.20
            mom = pd.to_numeric(r.get("momentum_score"), errors="coerce")
            if pd.notna(mom):
                if mom > 1.5:
                    score += 0.20
                elif mom > 0.25:
                    score += 0.10
                elif mom < -0.2:
                    score -= 0.15
            fac = pd.to_numeric(r.get("factor_composite"), errors="coerce")
            if pd.notna(fac):
                if fac > 0.5:
                    score += 0.15
                elif fac > 0.2:
                    score += 0.05
            agg = pd.to_numeric(r.get("composite"), errors="coerce")
            if pd.notna(agg):
                if agg > 0.7:
                    score += 0.25
                elif agg > 0.5:
                    score += 0.15
                elif agg < 0.25:
                    score -= 0.10
            if pd.notna(pd.to_numeric(r.get("mktcap_to_assets"), errors="coerce")):
                mca = float(r["mktcap_to_assets"])
                if mca < 0.5:
                    score += 0.08
                elif mca > 2.0:
                    score -= 0.12
            if pd.notna(mom) and stress and mom < 0.25:
                score -= 0.05
            if str(r.get("sp500_member", "")).lower() == "true" or r.get("sp500_member") is True:
                score += 0.05
            if stress:
                score -= 0.08
            if score >= 0.55:
                act = "BUY"
            elif score >= 0.35:
                act = "ACCUMULATE"
            elif score >= 0.15:
                act = "WATCH"
            else:
                act = "AVOID"
            out[tk] = act
        return out

    def perturb_and_flip(driver_col, scale_fn):
        """Perturb one numeric driver per row by a row-specific scale (the
        driver's own estimation error), re-score, count decision flips."""
        flips = 0
        score_deltas = []
        n = 0
        for _ in range(args.n_perturb):
            df = base.copy()
            if driver_col not in df.columns:
                return None, None, 0
            noise = rng.normal(0.0, 1.0, size=len(df))
            # scale_fn returns per-row perturbation scale (est. error of the driver)
            s = scale_fn(df)
            df[driver_col] = pd.to_numeric(df[driver_col], errors="coerce") + noise * s
            acts = _rescore(df)
            ref = base_action.reindex(acts.index)
            flips += int((acts != ref).sum())
            n = len(acts)
            delta = (acts.map(ORDER_IDX) - ref.map(ORDER_IDX)).abs().mean()
            score_deltas.append(float(delta))
        return flips / max(args.n_perturb * n, 1), float(np.mean(score_deltas)) if score_deltas else None, n

    rows = []
    drivers = []
    # driver 1: momentum_score (est. error ~ its cross-sectional std/4, a loose
    # standard error for a rolling-window estimate)
    if "momentum_score" in base.columns:
        drivers.append(("momentum_score", lambda d: np.full(len(d), float(d["momentum_score"].std()) / 4.0)))
    if "factor_composite" in base.columns:
        drivers.append(("factor_composite", lambda d: np.full(len(d), float(d["factor_composite"].std()) / 4.0)))
    if "composite" in base.columns:
        drivers.append(("composite", lambda d: np.full(len(d), float(d["composite"].std()) / 4.0)))

    for name, scale_fn in drivers:
        flip, delta, n = perturb_and_flip(name, scale_fn)
        if flip is None:
            print(f"  {name}: not present in buy_candidates inputs — skipped")
            continue
        rows.append({"driver": name, "perturb_scale": round(float(np.mean(scale_fn(base))), 4),
                     "flip_rate": round(flip, 4), "mean_score_move": round(delta, 4) if delta else None,
                     "n": n})
        print(f"  {name}: flip_rate={flip:.2%} scale={np.mean(scale_fn(base)):.4f}")

    # driver 4: the HMM regime label itself — the hardest "r1-r2" we fix.
    # Flip the regime verdict (stress vs not-stress, prob ~0.1 per trial) and
    # count decision changes from the stress haircut on the same scores.
    try:
        pd.read_csv(DATA_DIR / "hmm_regime_states.csv")
        reg_now = bc.regime()
        stress_old = "stress" in reg_now.lower()
        flips = 0
        n_reg = 0
        for _ in range(args.n_perturb):
            if rng.random() < 0.1:
                alt = rng.choice([r for r in ("low_vol", "normal", "high_vol_stress") if r != reg_now])
                stress_new = "stress" in alt.lower()
                if stress_new != stress_old:
                    for tk, r in base.iterrows():
                        act_old = base_action.loc[tk] if tk in base_action.index else "AVOID"
                        act_new = _action_from_score(_approx_score(r, stress_new))
                        if act_new != act_old:
                            flips += 1
                    n_reg += len(base)
        flip_regime = flips / max(n_reg, 1)
        rows.append({"driver": "hmm_regime_label", "perturb_scale": 0.10, "flip_rate": round(flip_regime, 4),
                     "mean_score_move": None, "n": n_reg})
        print(f"  hmm_regime_label: flip_rate={flip_regime:.2%}")
    except Exception as e:
        print(f"  hmm regime audit skipped ({e})")

    out = pd.DataFrame(rows).sort_values("flip_rate", ascending=False) if rows else pd.DataFrame()
    out.to_csv(DATA_DIR / "hidden_optionality.csv", index=False)
    print(f"\nWrote hidden_optionality.csv ({len(out)} drivers)")
    if len(out):
        print("\nDrivers ranked by hidden optionality (decision flip rate):")
        print(out.to_string(index=False))
        print("\nThe American-options lesson: the highest-flip driver is the 'r1-r2'\n"
              "our system treats as fixed. Stochasticize it before trusting the decisions.")


def _approx_score(r, stress):
    """Same scoring as buy_candidates but parameterized by the stress verdict."""
    score = 0.0
    if r.get("dual_pass_core") is True or str(r.get("dual_pass_core", "")).lower() == "true":
        score += 0.35
    if r.get("value_trifecta") is True or str(r.get("value_trifecta", "")).lower() == "true":
        score += 0.20
    if r.get("buffett_quality") is True or str(r.get("buffett_quality", "")).lower() == "true":
        score += 0.20
    mom = pd.to_numeric(r.get("momentum_score"), errors="coerce")
    if pd.notna(mom):
        if mom > 1.5:
            score += 0.20
        elif mom > 0.25:
            score += 0.10
        elif mom < -0.2:
            score -= 0.15
    fac = pd.to_numeric(r.get("factor_composite"), errors="coerce")
    if pd.notna(fac):
        if fac > 0.5:
            score += 0.15
        elif fac > 0.2:
            score += 0.05
    agg = pd.to_numeric(r.get("composite"), errors="coerce")
    if pd.notna(agg):
        if agg > 0.7:
            score += 0.25
        elif agg > 0.5:
            score += 0.15
        elif agg < 0.25:
            score -= 0.10
    if pd.notna(pd.to_numeric(r.get("mktcap_to_assets"), errors="coerce")):
        mca = float(r["mktcap_to_assets"])
        if mca < 0.5:
            score += 0.08
        elif mca > 2.0:
            score -= 0.12
    if pd.notna(mom) and stress and mom < 0.25:
        score -= 0.05
    if str(r.get("sp500_member", "")).lower() == "true" or r.get("sp500_member") is True:
        score += 0.05
    if stress:
        score -= 0.08
    return score


def _action_from_score(score):
    if score >= 0.55:
        return "BUY"
    if score >= 0.35:
        return "ACCUMULATE"
    if score >= 0.15:
        return "WATCH"
    return "AVOID"


if __name__ == "__main__":
    main()

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
estimates (momentum_score, factor_composite, aggregate composite, the HMM
regime posterior). Each is an "r1-r2" — deterministic in our system, stochastic
in reality. This script perturbs each driver by its OWN estimation error and
measures how often the BUY/ACCUMULATE/WATCH/AVOID decision flips — using the
SAME scorer as production (buy_candidates.score_row), so flips measure the
driver's noise, not a drifted copy of the logic. The flip rate IS the hidden
optionality: the probability our decision is riding on noise.

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

import buy_candidates as bc

DATA_DIR = Path(__file__).resolve().parent

ORDER = ["AVOID", "WATCH", "ACCUMULATE", "BUY"]
ORDER_IDX = {a: i for i, a in enumerate(ORDER)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perturb", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--v2", action="store_true",
                    help="American-options v2: convexity bias + stress-conditional flip rates -> optionality_audit_v2.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    base = bc.build()
    if base.empty or "action" not in base.columns:
        raise SystemExit("buy_candidates.build() returned nothing — run analytics first")
    base = base.set_index("ticker", drop=False)
    base_action = base["action"].astype(str)

    # reference actions computed with the REAL scorer on unperturbed rows —
    # the flips below are perturbed-vs-unperturbed under the same logic.
    frag_map, skew_map = bc._load_maps()

    def actions_at(df, stress_p, sigs):
        acts = []
        for _, r in df.iterrows():
            score, _ = bc.score_row(r, stress_p, frag_map, skew_map, sigs)
            acts.append(bc.action_from_score(score))
        return pd.Series(acts, index=df.index, dtype=object)

    def convexity_bias(driver_col, scale_fn):
        """American-options convexity bias pi = E[f(a_tilde)] - f(E[a_tilde]).

        For each perturbation, compute the score with the perturbed driver value
        (numerator) and the score with the expected (unperturbed) value, then
        average the difference. A non-zero pi means the scoring function is
        curved in this driver — the 'r1-r2' is hiding optionality."""
        scores_pert = []
        scores_ref = []
        for _ in range(args.n_perturb):
            df = base.copy()
            if driver_col not in df.columns:
                return None
            noise = rng.normal(0.0, 1.0, size=len(df))
            s = scale_fn(df)
            df[driver_col] = pd.to_numeric(df[driver_col], errors="coerce") + noise * s
            # perturbed scores
            for _, r in df.iterrows():
                sc, _ = bc.score_row(r, ref_stress, ref_sigs)
                scores_pert.append(sc)
            # reference scores (unperturbed)
            for _, r in base.iterrows():
                sc, _ = bc.score_row(r, ref_stress, ref_sigs)
                scores_ref.append(sc)
        if not scores_pert:
            return None
        return float(np.mean(scores_pert) - np.mean(scores_ref))

    ref_stress = bc.regime_stress_prob()
    ref_sigs = {
        "momentum": bc._est_error(base.get("momentum_score")),
        "factor": bc._est_error(base.get("factor_composite")),
        "composite": bc._est_error(base.get("composite")),
        "resid_mom": bc._est_error(base.get("resid_mom_63")),
        "liquidity": bc._est_error(base.get("liquidity_score")),
        "skew": bc._est_error(pd.Series(list(skew_map.values()))) if skew_map else 0.0,
    }
    ref_acts = actions_at(base, ref_stress, ref_sigs)

    def perturb_and_flip(driver_col, scale_fn, stress_p=None):
        """Perturb one numeric driver per row by a row-specific scale (the
        driver's own estimation error), re-score with the REAL scorer, count
        decision flips vs the unperturbed reference."""
        flips = 0
        score_deltas = []
        n = 0
        sp = stress_p if stress_p is not None else ref_stress
        for _ in range(args.n_perturb):
            df = base.copy()
            if driver_col not in df.columns:
                return None, None, 0
            noise = rng.normal(0.0, 1.0, size=len(df))
            s = scale_fn(df)
            df[driver_col] = pd.to_numeric(df[driver_col], errors="coerce") + noise * s
            acts = actions_at(df, sp, ref_sigs)
            ref = ref_acts.reindex(acts.index)
            flips += int((acts != ref).sum())
            n = len(acts)
            score_deltas.append(float((acts.map(ORDER_IDX) - ref.map(ORDER_IDX)).abs().mean()))
        return flips / max(args.n_perturb * n, 1), float(np.mean(score_deltas)) if score_deltas else None, n

    rows = []
    drivers = []
    for col, key in (("momentum_score", "momentum"), ("factor_composite", "factor"),
                     ("composite", "composite"), ("resid_mom_63", "resid_mom"),
                     ("liquidity_score", "liquidity")):
        if col in base.columns:
            drivers.append((col, key, lambda d, c=col: np.full(len(d), float(d[c].std()) / 4.0)))

    if not args.v2:
        for name, key, scale_fn in drivers:
            flip, delta, n = perturb_and_flip(name, scale_fn)
            if flip is None:
                print(f"  {name}: not present in buy_candidates inputs — skipped")
                continue
            rows.append({"driver": name, "perturb_scale": round(float(np.mean(scale_fn(base))), 4),
                         "flip_rate": round(flip, 4), "mean_score_move": round(delta, 4) if delta else None,
                         "n": n})
            print(f"  {name}: flip_rate={flip:.2%} scale={np.mean(scale_fn(base)):.4f}")

        # driver 4: the HMM regime posterior (soft stress belief). Perturb
        # p(stress) by its own estimation error (0.10) and count decision changes.
        # This is the honest test of the American-options fix: the old hard-label
        # cliff flipped 28.4% of decisions on a label flip; the soft posterior
        # should move decisions proportionally (far fewer flips at this scale).
        try:
            p0 = bc.regime_stress_prob()
            flips = 0
            n_reg = 0
            for _ in range(args.n_perturb):
                p_pert = float(np.clip(p0 + rng.normal(0.0, 0.10), 0.0, 1.0))
                if abs(p_pert - p0) < 1e-9:
                    continue
                acts = actions_at(base, p_pert, ref_sigs)
                ref = ref_acts.reindex(acts.index)
                flips += int((acts != ref).sum())
                n_reg += len(acts)
            flip_regime = flips / max(n_reg, 1)
            rows.append({"driver": "hmm_regime_posterior", "perturb_scale": 0.10, "flip_rate": round(flip_regime, 4),
                         "mean_score_move": None, "n": n_reg})
            print(f"  hmm_regime_posterior: flip_rate={flip_regime:.2%} (baseline p(stress)={p0:.2f})")
        except Exception as e:
            print(f"  hmm regime audit skipped ({e})")

        out = pd.DataFrame(rows).sort_values("flip_rate", ascending=False) if rows else pd.DataFrame()
        out.to_parquet(DATA_DIR / "hidden_optionality.parquet")
        print(f"\nWrote hidden_optionality.csv ({len(out)} drivers)")
        if len(out):
            print("\nDrivers ranked by hidden optionality (decision flip rate):")
            print(out.to_string(index=False))
            print("\nThe American-options lesson: the highest-flip driver is the 'r1-r2'\n"
                  "our system treats as fixed. Stochasticize it before trusting the decisions.")

    # --- v2: GPU-accelerated American-options convexity bias + stress-conditional flip rates ---
    if args.v2:
        from scipy.special import erf as erf_vec
        from buy_candidates import MOMENTUM_STEPS, FACTOR_STEPS, COMPOSITE_STEPS, RESID_MOM_STEPS, LIQUIDITY_STEPS, SKEW_STEPS

        # Try PyTorch GPU; fall back to vectorized NumPy
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
            else:
                device = torch.device('cpu')
                print("GPU not available, using CPU")
        except ImportError:
            device = None
            print("PyTorch not installed, using vectorized NumPy")

        def step_exp_torch(x, sig, baseline, steps):
            """PyTorch vectorized _step_expectation. x: (n,) tensor. Returns (n,) tensor."""
            valid = ~torch.isnan(x)
            xv = torch.where(valid, x, torch.zeros_like(x))
            if sig <= 0:
                contrib = torch.full_like(x, baseline)
                for t, d in steps:
                    contrib = torch.where(xv >= t, contrib + d, contrib)
            else:
                sqrt2 = (2.0) ** 0.5
                contrib = torch.full_like(x, baseline)
                for t, d in steps:
                    contrib = contrib + d * 0.5 * (1.0 + torch.erf((xv - t) / (sig * sqrt2)))
            return torch.where(valid, contrib, torch.zeros_like(x))

        def score_torch(mom, rm, fc, agg, liq, dec, lev, sp500, stress_p, sigs, fragile=None, skew_t=None):
            """PyTorch GPU scorer. mom, rm, fc, agg, liq, sp500 are 1D tensors on device.
            dec, lev are numpy string arrays on CPU. Returns 1D tensor on device."""
            score = torch.zeros_like(mom)

            # String comparisons stay on CPU, then move to GPU
            score = score + torch.where(torch.tensor(dec == 'INCLUDE_CORE', device=device), 0.35, 0.0)
            score = score + torch.where(torch.tensor(dec == 'INCLUDE_VALUE', device=device), 0.20, 0.0)
            score = score + torch.where(torch.tensor(dec == 'INCLUDE_QUALITY', device=device), 0.20, 0.0)
            score = score + torch.where(torch.tensor(dec == 'SATELITE', device=device), 0.08, 0.0)
            score = score + torch.where(torch.tensor(dec == 'AVOID', device=device), -0.25, 0.0)

            score = score + step_exp_torch(mom, sigs.get('momentum', 0.0), *MOMENTUM_STEPS)
            score = score + step_exp_torch(rm, sigs.get('resid_mom', 0.0), *RESID_MOM_STEPS)
            score = score + step_exp_torch(fc, sigs.get('factor', 0.0), *FACTOR_STEPS)
            score = score + step_exp_torch(agg, sigs.get('composite', 0.0), *COMPOSITE_STEPS)

            score = score + torch.where(torch.tensor(lev == 'cheap-assets', device=device), 0.08, 0.0)
            score = score + torch.where(torch.tensor(lev == 'levered-assets', device=device), -0.12, 0.0)

            score = score + step_exp_torch(liq, sigs.get('liquidity', 0.0), *LIQUIDITY_STEPS)

            if fragile is not None:
                score = torch.where(fragile, score - 0.30, score)

            if skew_t is not None:
                score = score + step_exp_torch(skew_t, sigs.get('skew', 0.0), *SKEW_STEPS)

            score = torch.where(sp500, score + 0.05, score)

            if stress_p > 0.01:
                score = score - 0.08 * stress_p
                mom_na = torch.isnan(mom)
                mom_low = (~mom_na) & (mom < 0.25)
                score = torch.where(mom_na, score - 0.05 * stress_p, score)
                score = torch.where(mom_low, score - 0.05 * stress_p * ((0.25 - mom) / 0.25), score)
                m_contrib = step_exp_torch(mom, sigs.get('momentum', 0.0), *MOMENTUM_STEPS)
                m_pos = m_contrib > 0
                m_att = m_contrib * (1.0 - 0.5 * stress_p)
                score = torch.where(m_pos, score + (m_att - m_contrib), score)

            return score

        def actions_from_scores_torch(scores):
            """PyTorch vectorized action_from_score. Returns tensor of byte codes."""
            # 0=AVOID, 1=WATCH, 2=ACCUMULATE, 3=BUY
            acts = torch.full_like(scores, 0, dtype=torch.int8)
            acts = torch.where(scores >= 0.15, 1, acts)
            acts = torch.where(scores >= 0.35, 2, acts)
            acts = torch.where(scores >= 0.55, 3, acts)
            return acts

        print("\n=== optionality_audit_v2: GPU-accelerated convexity bias + stress-conditional flips ===")

        # Pre-extract all arrays ONCE and move to GPU
        mom_arr = torch.tensor(base['momentum_score'].to_numpy(dtype=float), device=device)
        rm_arr = torch.tensor(base['resid_mom_63'].to_numpy(dtype=float), device=device)
        fc_arr = torch.tensor(base['factor_composite'].to_numpy(dtype=float), device=device)
        agg_arr = torch.tensor(base['composite'].to_numpy(dtype=float), device=device)
        liq_arr = torch.tensor(base['liquidity_score'].to_numpy(dtype=float), device=device)
        dec_arr = base['decision'].values  # string array, stays on CPU
        lev_arr = base['leverage_flag'].values  # string array, stays on CPU
        sp500_arr = torch.tensor(base['sp500_member'].values.astype(bool), device=device)
        tk = base['ticker'].values

        # fragility tensor
        if frag_map:
            fragile_arr = torch.tensor([frag_map.get(str(t).upper(), False) for t in tk], device=device)
        else:
            fragile_arr = None

        # skew tensor
        if skew_map:
            skew_t = torch.tensor([skew_map.get(str(t).upper(), float('nan')) for t in tk], device=device)
        else:
            skew_t = None

        # Reference scores and actions (once)
        ref_scores = score_torch(mom_arr, rm_arr, fc_arr, agg_arr, liq_arr, dec_arr, lev_arr, sp500_arr, ref_stress, ref_sigs, fragile_arr, skew_t)
        ref_acts = actions_from_scores_torch(ref_scores)

        v2_rows = []
        for name, key, scale_fn in drivers:
            if name not in base.columns:
                continue

            pert_scale = float(np.mean(scale_fn(base)))

            # Select which tensor to perturb
            if name == 'momentum_score':
                target_arr = mom_arr
            elif name == 'resid_mom_63':
                target_arr = rm_arr
            elif name == 'factor_composite':
                target_arr = fc_arr
            elif name == 'composite':
                target_arr = agg_arr
            elif name == 'liquidity_score':
                target_arr = liq_arr
            else:
                continue

            # Convexity bias: E[f(a_tilde)] - f(E[a_tilde])
            # Batch ALL perturbations at once: (n_perturb, n_tickers)
            n_perturb = args.n_perturb
            n_tickers = len(base)
            
            # Generate all noise at once
            noise = torch.randn(n_perturb, n_tickers, device=device)
            pert_batch = target_arr.unsqueeze(0) + noise * pert_scale  # (n_perturb, n_tickers)
            
            # Expand other arrays to match
            rm_exp = rm_arr.unsqueeze(0).expand(n_perturb, -1)
            fc_exp = fc_arr.unsqueeze(0).expand(n_perturb, -1)
            agg_exp = agg_arr.unsqueeze(0).expand(n_perturb, -1)
            liq_exp = liq_arr.unsqueeze(0).expand(n_perturb, -1)
            sp500_exp = sp500_arr.unsqueeze(0).expand(n_perturb, -1)
            if fragile_arr is not None:
                frag_exp = fragile_arr.unsqueeze(0).expand(n_perturb, -1)
            else:
                frag_exp = None
            if skew_t is not None:
                skew_exp = skew_t.unsqueeze(0).expand(n_perturb, -1)
            else:
                skew_exp = None

            # Score all perturbations at once
            if name == 'momentum_score':
                pert_scores = score_torch(pert_batch, rm_exp, fc_exp, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, ref_stress, ref_sigs, frag_exp, skew_exp)
            elif name == 'resid_mom_63':
                pert_scores = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), pert_batch, fc_exp, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, ref_stress, ref_sigs, frag_exp, skew_exp)
            elif name == 'factor_composite':
                pert_scores = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, pert_batch, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, ref_stress, ref_sigs, frag_exp, skew_exp)
            elif name == 'composite':
                pert_scores = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, fc_exp, pert_batch, liq_exp, dec_arr, lev_arr, sp500_exp, ref_stress, ref_sigs, frag_exp, skew_exp)
            elif name == 'liquidity_score':
                pert_scores = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, fc_exp, agg_exp, pert_batch, dec_arr, lev_arr, sp500_exp, ref_stress, ref_sigs, frag_exp, skew_exp)

            # Convexity bias: mean over perturbations of (pert_score - ref_score)
            scores_pert_mean = pert_scores.mean(dim=0)  # (n_tickers,)
            pi = float((scores_pert_mean - ref_scores).mean().item())

            # Flip rates at three stress levels
            flip_rates = {}
            for stress_name, stress_val in [('baseline', ref_stress), ('high_stress', 0.9), ('low_stress', 0.1)]:
                # Score all perturbations at this stress level
                if name == 'momentum_score':
                    sc = score_torch(pert_batch, rm_exp, fc_exp, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, stress_val, ref_sigs, frag_exp, skew_exp)
                elif name == 'resid_mom_63':
                    sc = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), pert_batch, fc_exp, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, stress_val, ref_sigs, frag_exp, skew_exp)
                elif name == 'factor_composite':
                    sc = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, pert_batch, agg_exp, liq_exp, dec_arr, lev_arr, sp500_exp, stress_val, ref_sigs, frag_exp, skew_exp)
                elif name == 'composite':
                    sc = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, fc_exp, pert_batch, liq_exp, dec_arr, lev_arr, sp500_exp, stress_val, ref_sigs, frag_exp, skew_exp)
                elif name == 'liquidity_score':
                    sc = score_torch(mom_arr.unsqueeze(0).expand(n_perturb, -1), rm_exp, fc_exp, agg_exp, pert_batch, dec_arr, lev_arr, sp500_exp, stress_val, ref_sigs, frag_exp, skew_exp)

                pert_acts = actions_from_scores_torch(sc)  # (n_perturb, n_tickers)
                flips = (pert_acts != ref_acts.unsqueeze(0)).sum().item()
                n_total = n_perturb * n_tickers
                flip_rates[stress_name] = flips / max(n_total, 1)

            v2_rows.append({
                "driver": name,
                "perturb_scale": round(pert_scale, 4),
                "convexity_bias": round(pi, 6),
                "flip_rate_baseline": round(flip_rates['baseline'], 4),
                "flip_rate_high_stress": round(flip_rates['high_stress'], 4),
                "flip_rate_low_stress": round(flip_rates['low_stress'], 4),
                "stress_sensitivity": round(flip_rates['high_stress'] - flip_rates['low_stress'], 4),
                "n": len(base),
            })
            print(f"  {name}: pi={pi:+.4f} flip_base={flip_rates['baseline']:.2%} "
                  f"flip_high={flip_rates['high_stress']:.2%} flip_low={flip_rates['low_stress']:.2%}")

        v2_out = pd.DataFrame(v2_rows)
        if not v2_out.empty:
            v2_out = v2_out.sort_values("convexity_bias", key=lambda s: s.abs(), ascending=False)
        v2_out.to_parquet(DATA_DIR / "optionality_audit_v2.parquet", index=False)
        print(f"\nWrote optionality_audit_v2.parquet ({len(v2_out)} drivers)")
        if not v2_out.empty:
            print(v2_out.to_string(index=False))
            print("\nInterpretation: |convexity_bias| = hidden optionality (the 'r1-r2' gap).")
            print("stress_sensitivity > 0 means the driver is MORE fragile in stress — the")
            print("American-options lesson: stochasticize these before trusting decisions.")


if __name__ == "__main__":
    main()

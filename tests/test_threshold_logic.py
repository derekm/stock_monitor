#!/usr/bin/env python3
"""Unit tests for threshold_logic.py"""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from threshold_logic import (
    BASE_THRESHOLDS, REGIME_THRESHOLDS, thresholds_for_regime, select_regime,
    failed_legs, is_dual_pass, distance_to_threshold, evaluate_universe,
)

def _row(**kw):
    base = dict(ticker="TEST", roe=0.15, roic=0.15, debt_to_equity=0.5,
                ev_ebitda=8.0, pb_ratio=1.2, mktcap_to_assets=0.3)
    base.update(kw)
    return pd.Series(base)

class TestThresholdLogic(unittest.TestCase):
    def test_base_keys(self):
        for k in ("roe_min","roic_min","de_max","ev_max","pb_max","mca_max"):
            self.assertIn(k, BASE_THRESHOLDS)

    def test_regimes(self):
        for r in ("low_vol","normal","high_vol_stress","uncertain"):
            self.assertIn(r, REGIME_THRESHOLDS)
            self.assertEqual(thresholds_for_regime(r)["roe_min"], 0.15)

    def test_stress_tighter(self):
        b, s = thresholds_for_regime("normal"), thresholds_for_regime("high_vol_stress")
        self.assertLessEqual(s["ev_max"], b["ev_max"])
        self.assertLessEqual(s["de_max"], b["de_max"])

    def test_low_vol_eases_value(self):
        b, lo = thresholds_for_regime("normal"), thresholds_for_regime("low_vol")
        self.assertGreaterEqual(lo["ev_max"], b["ev_max"])
        self.assertEqual(lo["roe_min"], b["roe_min"])

    def test_uncertain_base(self):
        b, u = thresholds_for_regime("normal"), thresholds_for_regime("uncertain")
        for k in BASE_THRESHOLDS:
            self.assertEqual(u[k], b[k])

    def test_dual_pass(self):
        self.assertTrue(is_dual_pass(_row()))
        self.assertEqual(failed_legs(_row(), BASE_THRESHOLDS), [])

    def test_fail_roic(self):
        self.assertIn("roic", failed_legs(_row(roic=0.10), BASE_THRESHOLDS))

    def test_fail_mca(self):
        self.assertIn("mca", failed_legs(_row(mktcap_to_assets=0.9), BASE_THRESHOLDS))

    def test_gaps(self):
        g = distance_to_threshold(_row(roe=0.20, roic=0.10), BASE_THRESHOLDS)
        self.assertGreater(g["roe_gap"], 0)
        self.assertLess(g["roic_gap"], 0)

    def test_select_uncertain(self):
        row = pd.Series({"regime":"normal","p_state_0":0.4,"p_state_1":0.3,"p_state_2":0.3})
        self.assertEqual(select_regime(row, soft_min=0.7), "uncertain")

    def test_select_confident(self):
        row = pd.Series({"regime":"high_vol_stress","p_state_0":0.05,"p_state_1":0.9,"p_state_2":0.05})
        self.assertEqual(select_regime(row, soft_min=0.7), "high_vol_stress")

    def test_evaluate(self):
        fund = pd.DataFrame([_row(ticker="A"), _row(ticker="B", roic=0.05)])
        out = evaluate_universe(fund, "normal")
        self.assertEqual(int(out.dual_pass.sum()), 1)

    def test_unknown_regime(self):
        self.assertEqual(thresholds_for_regime("xyz")["label"], REGIME_THRESHOLDS["normal"]["label"])

if __name__ == "__main__":
    unittest.main()

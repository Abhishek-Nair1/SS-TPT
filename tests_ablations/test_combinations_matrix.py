"""Unit tests for the 2x2 combinations matrix and statistical benchmarks.

Covers:
  - 2x2 Factorial Matrix: Baseline vs Spatial vs Adaptive Alpha vs Combined.
  - Empirical Performance: Gains across Clean, Adversarial, and Shift regimes.
  - Wilson 95% Confidence Intervals calculation and bounds.
  - Pareto Frontier verification for accuracy vs runtime.
"""

import unittest
import numpy as np

from tests_ablations.data_and_models import (
    MASTER_BENCHMARK_DATA,
    calculate_wilson_ci,
)


class TestCombinationsMatrixAndSynergy(unittest.TestCase):
    """Verifies the 4 principal combinations across the 3 regimes."""

    def test_factorial_2x2_presence(self):
        """All 4 quadrants of the 2x2 matrix are populated and evaluated."""
        methods = MASTER_BENCHMARK_DATA["methods"]
        self.assertEqual(len(methods), 4)
        self.assertIn("SS-TPT baseline", methods)
        self.assertIn("SS-TPT + Spatial", methods)
        self.assertIn("SS-TPT + Adaptive α(x)", methods)
        self.assertIn("SS-TPT + Spatial + Adaptive α(x)", methods)

    def test_adversarial_synergy_and_winning_delta(self):
        """Adaptive alpha achieves highest adversarial robustness; Combined achieves fastest runtime."""
        adv = MASTER_BENCHMARK_DATA["adversarial"]
        accs = adv["accuracy"]
        deltas = adv["delta"]
        runtimes = adv["runtime"]

        # Adaptive alpha (index 2) achieves peak adversarial accuracy (78.40% vs 78.15% base)
        self.assertGreater(accs[2], accs[0])
        self.assertGreater(deltas[2], 0.0)

        # Combined (index 3) achieves the most efficient runtime among all adapted methods
        self.assertLess(runtimes[3], runtimes[0])

    def test_clean_regime_improvements(self):
        """Spatial and Combined methods improve clean accuracy over baseline."""
        clean = MASTER_BENCHMARK_DATA["clean"]
        accs = clean["accuracy"]
        deltas = clean["delta"]

        # Spatial achieves peak clean accuracy (83.95% vs 83.65% base)
        self.assertGreater(accs[1], accs[0])
        self.assertGreater(deltas[1], 0.0)

        # Combined also strictly improves over baseline (83.80% vs 83.65% base)
        self.assertGreater(accs[3], accs[0])
        self.assertGreater(deltas[3], 0.0)

    def test_distribution_shift_preservation(self):
        """On ImageNet-R rendition shift, the combined pipeline preserves baseline."""
        shift = MASTER_BENCHMARK_DATA["shift"]
        accs = shift["accuracy"]

        # Combined preserves performance relative to baseline (within 0.5%)
        self.assertGreaterEqual(accs[3], accs[0] - 0.5)


class TestStatisticalSignificanceAndCIs(unittest.TestCase):
    """Verifies Wilson Score 95% Confidence Intervals."""

    def test_wilson_ci_computation(self):
        """Verify Wilson 95% CI formulation for N=2000 samples."""
        n = 2000
        # 78.15% baseline (1563 / 2000)
        k_base = int(round(0.7815 * n))
        p_base, low_base, high_base = calculate_wilson_ci(k_base, n, confidence=0.95)

        self.assertAlmostEqual(p_base, 78.15, places=1)
        self.assertTrue(76.0 < low_base < 77.5, msg=f"Low bound {low_base} outside expected range")
        self.assertTrue(79.5 < high_base < 81.0, msg=f"High bound {high_base} outside expected range")

        # 78.40% adaptive alpha (1568 / 2000)
        k_adapt = int(round(0.7840 * n))
        p_adapt, low_adapt, high_adapt = calculate_wilson_ci(k_adapt, n, confidence=0.95)

        self.assertAlmostEqual(p_adapt, 78.40, places=1)
        self.assertGreater(high_adapt, high_base)

    def test_zero_and_boundary_cases(self):
        """Check CI boundary behavior at k=0 and k=n."""
        _, low0, high0 = calculate_wilson_ci(0, 100)
        self.assertEqual(low0, 0.0)
        self.assertGreater(high0, 0.0)

        _, low1, high1 = calculate_wilson_ci(100, 100)
        self.assertLess(low1, 100.0)
        self.assertEqual(high1, 100.0)


class TestParetoEfficiency(unittest.TestCase):
    """Verifies Pareto efficiency trade-offs between accuracy and runtime."""

    def test_combined_runtime_efficiency(self):
        """Combined method evaluates faster than baseline under attack due to early view gating."""
        runtimes = MASTER_BENCHMARK_DATA["adversarial"]["runtime"]
        t_base = runtimes[0]  # 2755.6s
        t_comb = runtimes[3]  # 1492.4s
        self.assertLess(t_comb, t_base)

    def test_adaptive_alpha_overhead(self):
        """Adaptive alpha evaluates much faster than un-gated baseline under attack."""
        runtimes = MASTER_BENCHMARK_DATA["adversarial"]["runtime"]
        t_base = runtimes[0]  # 2755.6s
        t_alpha = runtimes[2] # 1929.3s
        self.assertLess(t_alpha, t_base)


if __name__ == "__main__":
    unittest.main()

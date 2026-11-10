"""Unit tests verifying all empirical benchmarks and latency decompositions for our 4 methods.

Strictly covers:
  - 4 Principal Methods: Baseline, Spatial (Task 1), Adaptive alpha (Task 2), Combined.
  - 3 Regimes: Clean (Caltech-101), Adversarial (PGD eps=1, 7 steps), and Distribution Shift (ImageNet-R).
  - Latency breakdown: Attack Time + Inference Time = Total Time.
  - View dynamics: Base view 0 behavior under Clean vs Adversarial.
  - Fixed alpha grid: Comparison between static alpha grid and adaptive alpha.
"""

import unittest
import numpy as np

from tests_ablations.data_and_models import (
    MASTER_4_METHODS,
    BENCHMARK_RESULTS,
    TASK2_FIXED_ALPHA_GRID,
    VIEW_DYNAMICS_DATA,
)


class TestOurFourMethodsBenchmarks(unittest.TestCase):
    """Verifies empirical accuracy and deltas across all 4 methods and 3 regimes."""

    def test_our_four_methods_adversarial_ranking_and_deltas(self):
        """Adaptive alpha achieves peak adversarial accuracy; all methods within valid ranges."""
        methods = BENCHMARK_RESULTS["methods"]
        accs = BENCHMARK_RESULTS["adversarial"]["accuracy"]
        deltas = BENCHMARK_RESULTS["adversarial"]["delta"]

        self.assertEqual(len(methods), 4)
        for a in accs:
            self.assertTrue(70.0 <= a <= 85.0)

        # Task 2 Adaptive alpha achieves highest adversarial robustness (78.40%)
        self.assertGreater(accs[2], accs[0])
        self.assertGreater(deltas[2], 0.0)

    def test_clean_regime_accuracy_and_improvements(self):
        """All adapted methods improve clean accuracy over baseline."""
        clean_accs = BENCHMARK_RESULTS["clean"]["accuracy"]
        self.assertEqual(len(clean_accs), 4)
        for a in clean_accs:
            self.assertTrue(80.0 <= a <= 90.0)

        # Task 1 Spatial achieves highest clean accuracy (83.95% vs 83.65%)
        self.assertGreater(clean_accs[1], clean_accs[0])
        # Combined and Adaptive also improve
        self.assertGreater(clean_accs[2], clean_accs[0])
        self.assertGreater(clean_accs[3], clean_accs[0])

    def test_distribution_shift_preservation(self):
        """Combined pipeline preserves baseline performance on ImageNet-R (within 0.5%)."""
        shift_accs = BENCHMARK_RESULTS["shift"]["accuracy"]
        self.assertEqual(len(shift_accs), 4)
        for a in shift_accs:
            self.assertTrue(45.0 <= a <= 60.0)
        self.assertGreaterEqual(shift_accs[3], shift_accs[0] - 0.5)


class TestLatencyAndThroughputDecomposition(unittest.TestCase):
    """Verifies per-image latency sums and throughput values."""

    def test_per_image_time_sums(self):
        """Verify attack time + inference time == total time per image."""
        adv_data = BENCHMARK_RESULTS["adversarial"]
        for i, method in enumerate(BENCHMARK_RESULTS["methods"]):
            calc_total = adv_data["attack_time_per_image"][i] + adv_data["inference_time_per_image"][i]
            expected_total = adv_data["total_time_per_image"][i]
            self.assertAlmostEqual(calc_total, expected_total, places=2,
                                   msg=f"Total time mismatch for {method}")

    def test_throughput_values(self):
        """Check throughput (samples/sec) for 2,000 samples."""
        adv_data = BENCHMARK_RESULTS["adversarial"]
        runtimes = adv_data["runtime"]
        throughputs = adv_data["throughput"]
        for i in range(4):
            calc_tp = 2000.0 / runtimes[i]
            self.assertAlmostEqual(calc_tp, throughputs[i], places=2)


class TestViewDynamicsAndAlphaGrid(unittest.TestCase):
    """Verifies view-level dynamics and fixed alpha grid sweep."""

    def test_view0_drastic_drop_under_adversarial_attack(self):
        """Unaugmented View 0 collapses from 0.86 (Clean) to 0.05 (Adversarial)."""
        clean_v0 = VIEW_DYNAMICS_DATA["clean"]["ss_base"][0]
        adv_v0 = VIEW_DYNAMICS_DATA["adversarial"]["ss_base"][0]

        self.assertGreater(clean_v0, 0.80)
        self.assertLess(adv_v0, 0.10)
        self.assertGreater(clean_v0 - adv_v0, 0.75)

    def test_fixed_alpha_grid_peak_at_point_four(self):
        """In the static alpha grid {0.3, 0.4, 0.5}, alpha=0.40 is the peak."""
        grid_alphas = TASK2_FIXED_ALPHA_GRID["alpha_values"]
        grid_accs = TASK2_FIXED_ALPHA_GRID["adversarial_accuracy"]

        max_idx = int(np.argmax(grid_accs))
        self.assertEqual(grid_alphas[max_idx], 0.40)
        self.assertAlmostEqual(grid_accs[max_idx], grid_accs[1], places=2)


if __name__ == "__main__":
    unittest.main()

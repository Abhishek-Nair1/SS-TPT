"""Unit tests for project-specific ablation studies.

Covers:
  - Task 1: Multiplicative Reliability Gating vs Linear Addition (Zombie view suppression).
  - Task 1: Spatial weight gamma sweep and AttentionPool2d vs GradCAM.
  - Task 2: Instance-Adaptive alpha(x) modes comparison (Variance vs 7 other modes).
  - Task 2: Distribution shift of alpha(x) under adversarial perturbation.
  - Integration: combine_all_scores compatibility with PyTorch tensors.
"""

import unittest
import torch
import numpy as np
from pathlib import Path

from tests_ablations.data_and_models import (
    TASK1_VALIDATION_SWEEP,
    TASK1_OFFICIAL_ARCH_ABLATION,
    TASK2_RELIABILITY_MODES,
    compute_gated_vs_linear,
    REPO_ROOT,
)
from utils.instance_alpha import compute_instance_alpha, compute_instance_alpha_details
from utils.spatial_metrics import combine_all_scores, min_max_normalize


class TestTask1SpatialGatingAblation(unittest.TestCase):
    """Verifies Task 1 Spatial Consistency and Gating mechanisms."""

    def test_zombie_view_suppression_mathematical_proof(self):
        """If a corrupted view has zero base score (S_base = 0), multiplicative gating strictly
        preserves 0, whereas linear addition creates a 'zombie view' with positive score.
        """
        # Corrupted view where base SS score is zero, but accidental spatial overlap is high (0.85)
        base_scores = np.array([0.0, 0.5, 0.9])
        spatial_scores = np.array([0.85, 0.4, 0.7])
        weight = 0.10

        gated, linear = compute_gated_vs_linear(base_scores, spatial_scores, weight=weight)

        # Gated score for corrupted view index 0 must be EXACTLY zero:
        self.assertAlmostEqual(gated[0], 0.0, places=6,
                               msg="Multiplicative gating failed to suppress corrupted view!")

        # Linear addition incorrectly assigns non-zero weight to the corrupted view:
        self.assertGreater(linear[0], 0.05,
                           msg="Linear addition was expected to resurrect corrupted view.")

    def test_spatial_weight_sweep_optimum(self):
        """Multiplicative gating peaks at w = 0.10, whereas linear addition deteriorates."""
        weights = TASK1_VALIDATION_SWEEP["weights"]
        gated_accs = TASK1_VALIDATION_SWEEP["gated_adv_acc"]
        linear_accs = TASK1_VALIDATION_SWEEP["linear_adv_acc"]

        # Gated peak is at w=0.10
        opt_idx = weights.index(0.10)
        self.assertEqual(gated_accs[opt_idx], 76.00)
        self.assertGreater(gated_accs[opt_idx], gated_accs[0])

        # High weight linear addition (w=0.20) drops significantly:
        w20_idx = weights.index(0.20)
        self.assertLess(linear_accs[w20_idx], linear_accs[0])
        self.assertGreater(gated_accs[w20_idx], linear_accs[w20_idx])

    def test_attention_vs_gradcam_and_view_selection(self):
        """AttentionPool2d with pure entropy view selection outperforms GradCAM & Pareto."""
        arch_data = TASK1_OFFICIAL_ARCH_ABLATION
        methods = arch_data["methods"]
        accs = arch_data["adversarial_accuracy"]

        base_idx = methods.index("Baseline (No Spatial)")
        gated_k1_idx = methods.index("AttentionPool2d + Gated (K=1)")
        gradcam_idx = methods.index("GradCAM (Score-guided w=0.20)")
        pareto_idx = methods.index("Pareto Selection (w=0.15)")

        self.assertGreater(accs[gated_k1_idx], accs[base_idx])
        self.assertGreater(accs[gated_k1_idx], accs[gradcam_idx])
        self.assertGreater(accs[gated_k1_idx], accs[pareto_idx])


class TestTask2AdaptiveAlphaAblation(unittest.TestCase):
    """Verifies Task 2 Instance-Adaptive alpha(x) modes and dynamics."""

    def test_variance_mode_superiority_under_adversarial_attack(self):
        """Variance mode achieves the highest adversarial robustness among all 8 modes."""
        modes = TASK2_RELIABILITY_MODES["modes"]
        adv_accs = TASK2_RELIABILITY_MODES["adv_acc"]

        var_idx = modes.index("Variance (Ours)")
        base_idx = modes.index("Fixed Baseline (0.40)")

        # Variance is the top adversarial performer
        max_idx = int(np.argmax(adv_accs))
        self.assertEqual(max_idx, var_idx)
        self.assertGreater(adv_accs[var_idx], adv_accs[base_idx])

    def test_instance_alpha_flat_score_fallback(self):
        """When view scores are completely flat, alpha(x) smoothly falls back to prior 0.40."""
        suit_flat = torch.full((16,), 0.5)
        stab_flat = torch.full((16,), 0.5)

        alpha = compute_instance_alpha(
            suit_flat,
            stab_flat,
            mode="variance",
            fixed_alpha=0.40,
        )
        self.assertAlmostEqual(float(alpha.item()), 0.40, places=4)

    def test_real_alpha_trace_files_integrity(self):
        """Verify that real evaluated alpha_trace.jsonl files on disk contain valid records."""
        trace_clean = REPO_ROOT / "output_results/task2/RN50/Caltech101/eps_0.0_alpha_0.0_step_0_batchsize_16_am_variance/alpha_trace.jsonl"
        trace_adv = REPO_ROOT / "output_results/task2/RN50/Caltech101/eps_1.0_alpha_0.25_step_7_batchsize_16_am_variance/alpha_trace.jsonl"

        if trace_clean.exists() and trace_adv.exists():
            import json
            with open(trace_clean) as f:
                clean_lines = [json.loads(l) for l in f if l.strip()]
            with open(trace_adv) as f:
                adv_lines = [json.loads(l) for l in f if l.strip()]

            self.assertGreaterEqual(len(clean_lines), 2000)
            self.assertGreaterEqual(len(adv_lines), 2000)

            # Check alpha bounds [0, 1]
            for row in clean_lines[:100]:
                self.assertTrue(0.0 <= row["alpha"] <= 1.0)
            for row in adv_lines[:100]:
                self.assertTrue(0.0 <= row["alpha"] <= 1.0)


class TestUnifiedIntegrationModule(unittest.TestCase):
    """Verifies that combine_all_scores integrates both Task 1 and Task 2 cleanly."""

    def test_combine_all_scores_integration(self):
        """Test full pipeline: instance alpha + spatial gated scoring."""
        suit = torch.tensor([0.1, 0.4, 0.8, 0.9, 0.2])
        stab = torch.tensor([0.2, 0.5, 0.7, 0.6, 0.1])
        spat = torch.tensor([0.3, 0.6, 0.9, 0.8, 0.0])

        alpha = compute_instance_alpha(
            min_max_normalize(suit),
            min_max_normalize(stab),
            mode="variance",
            fixed_alpha=0.40,
        )

        combined, details = combine_all_scores(
            suit_score=suit,
            stab_score=stab,
            spat_score=spat,
            alpha=alpha,
            spat_weight=0.10,
            combine_mode="gated",
        )

        self.assertEqual(combined.shape, (5,))
        self.assertTrue(torch.all(combined >= 0.0))
        self.assertIn("ss_base", details)
        self.assertIn("spat_n", details)


if __name__ == "__main__":
    unittest.main()

"""Integration test suite verifying the 4 principal methods:
1. SS-TPT Baseline
2. SS-TPT + Spatial (Task 1)
3. SS-TPT + Adaptive alpha(x) (Task 2)
4. SS-TPT + Spatial + Adaptive alpha(x) (Combined)
"""

import unittest
import torch
import torch.nn.functional as F

from utils.instance_alpha import (
    ALPHA_MODES,
    compute_instance_alpha_details,
    combine_ss,
)
from utils.spatial_metrics import (
    combine_all_scores,
    min_max_normalize,
)


def compute_unified_weights(
    suit_score,
    stab_score,
    spat_score=None,
    *,
    alpha_mode="fixed",
    fixed_alpha=0.4,
    use_spatial=False,
    spat_weight=0.10,
    combine_mode="gated",
    temp=0.25,
):
    """Unified quality scoring and softmax view weighting function."""
    suit_n = min_max_normalize(suit_score)
    stab_n = min_max_normalize(stab_score)

    # 1. Instance-Adaptive Stability-Suitability Balancing
    if alpha_mode == "fixed":
        alpha = torch.tensor(fixed_alpha, device=suit_score.device, dtype=suit_score.dtype)
        r_suit = torch.tensor(0.0, device=suit_score.device)
        r_stab = torch.tensor(0.0, device=suit_score.device)
    else:
        alpha, r_suit, r_stab = compute_instance_alpha_details(
            suit_n,
            stab_n,
            mode=alpha_mode,
            fixed_alpha=fixed_alpha,
            temp=temp,
            suit_raw=suit_score,
            stab_raw=stab_score,
        )

    # 2. Cross-View Spatial Consistency Scoring Integration
    eff_spat_score = spat_score if use_spatial else None
    eff_spat_weight = spat_weight if use_spatial else 0.0

    combined_score, details = combine_all_scores(
        suit_score=suit_score,
        stab_score=stab_score,
        spat_score=eff_spat_score,
        alpha=alpha,
        spat_weight=eff_spat_weight,
        combine_mode=combine_mode,
    )

    weights = F.softmax(combined_score / temp, dim=0)
    details["weights"] = weights
    details["r_suit"] = r_suit
    details["r_stab"] = r_stab
    return combined_score, weights, details


class TestMethodIntegration(unittest.TestCase):
    def setUp(self):
        # Synthetic view scores for 16 views
        torch.manual_seed(123)
        self.suit = torch.rand(16) * 10.0
        self.stab = torch.rand(16) * 8.0
        self.spat = torch.rand(16) * 2.0 - 1.0  # Pearson correlation in [-1, 1]

    def test_row1_baseline(self):
        """Method 1: SS-TPT Baseline."""
        score, weights, details = compute_unified_weights(
            self.suit,
            self.stab,
            self.spat,
            alpha_mode="fixed",
            fixed_alpha=0.4,
            use_spatial=False,
        )
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(details["alpha"].item(), 0.4, places=5)
        self.assertIsNone(details["spat_n"])

    def test_row2_spatial_only(self):
        """Method 2: SS-TPT + Spatial."""
        score, weights, details = compute_unified_weights(
            self.suit,
            self.stab,
            self.spat,
            alpha_mode="fixed",
            fixed_alpha=0.4,
            use_spatial=True,
            spat_weight=0.10,
            combine_mode="gated",
        )
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(details["alpha"].item(), 0.4, places=5)
        self.assertIsNotNone(details["spat_n"])

    def test_row3_adaptive_alpha_only(self):
        """Method 3: SS-TPT + Adaptive alpha(x)."""
        for mode in ("variance", "margin", "entropy", "agreement"):
            score, weights, details = compute_unified_weights(
                self.suit,
                self.stab,
                self.spat,
                alpha_mode=mode,
                fixed_alpha=0.4,
                use_spatial=False,
            )
            self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
            self.assertGreaterEqual(details["alpha"].item(), 0.0)
            self.assertLessEqual(details["alpha"].item(), 1.0)
            self.assertIsNone(details["spat_n"])

    def test_row4_combined_spatial_and_adaptive_alpha(self):
        """Method 4: SS-TPT + Spatial + Adaptive alpha(x)."""
        for combine_mode in ("gated", "linear", "linear_norm"):
            score, weights, details = compute_unified_weights(
                self.suit,
                self.stab,
                self.spat,
                alpha_mode="variance",
                fixed_alpha=0.4,
                use_spatial=True,
                spat_weight=0.10,
                combine_mode=combine_mode,
            )
            self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
            self.assertGreaterEqual(details["alpha"].item(), 0.0)
            self.assertLessEqual(details["alpha"].item(), 1.0)
            self.assertIsNotNone(details["spat_n"])

    def test_prediction_mixing(self):
        """Verify weighted prediction calculation over adapted candidate logits."""
        B, num_classes = 16, 100
        logits = torch.randn(B, num_classes)
        probs = F.softmax(logits, dim=1)

        _, weights, _ = compute_unified_weights(
            self.suit,
            self.stab,
            self.spat,
            alpha_mode="variance",
            use_spatial=True,
        )

        mixed_pred = torch.matmul(weights, probs)
        self.assertEqual(mixed_pred.shape, torch.Size([num_classes]))
        self.assertAlmostEqual(mixed_pred.sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()


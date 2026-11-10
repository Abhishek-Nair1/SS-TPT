"""CPU unit tests for spatial consistency scoring and unified score combination."""

import unittest
import torch
import torch.nn.functional as F

from utils.spatial_metrics import (
    TinyAugmenterWithTransform,
    compute_pearson_spatial_similarity,
    min_max_normalize,
    combine_all_scores,
)


class TestSpatialMetrics(unittest.TestCase):
    def test_tiny_augmenter_and_align(self):
        torch.manual_seed(42)
        augmenter = TinyAugmenterWithTransform()
        x = torch.rand(4, 3, 224, 224)
        x_aug, params = augmenter(x)

        self.assertEqual(x_aug.shape, x.shape)
        self.assertIn("angle", params)
        self.assertIn("translate", params)
        self.assertIn("scale", params)

        spatial_map = torch.rand(4, 1, 224, 224)
        aligned_map = augmenter.align_spatial_map(spatial_map, params)
        self.assertEqual(aligned_map.shape, spatial_map.shape)

    def test_pearson_similarity_identical(self):
        # A patterned heatmap identical to itself
        base = torch.zeros(2, 1, 64, 64)
        base[:, :, 20:44, 20:44] = 1.0  # Center activation box
        sim = compute_pearson_spatial_similarity(base, base, crop_margin=4)
        self.assertEqual(sim.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(sim, torch.tensor([1.0, 1.0]), atol=1e-5))

    def test_pearson_similarity_inverted(self):
        # Inverted activation
        base = torch.zeros(2, 1, 64, 64)
        base[:, :, 20:44, 20:44] = 1.0
        inverted = 1.0 - base
        sim = compute_pearson_spatial_similarity(base, inverted, crop_margin=4)
        self.assertTrue(torch.allclose(sim, torch.tensor([-1.0, -1.0]), atol=1e-5))

    def test_min_max_normalize(self):
        flat = torch.full((5,), 4.0)
        norm_flat = min_max_normalize(flat)
        self.assertTrue(torch.allclose(norm_flat, torch.zeros(5)))

        vec = torch.tensor([1.0, 2.0, 3.0, 5.0])
        norm_vec = min_max_normalize(vec)
        self.assertAlmostEqual(norm_vec[0].item(), 0.0)
        self.assertAlmostEqual(norm_vec[-1].item(), 1.0)
        self.assertGreaterEqual(norm_vec.min().item(), 0.0)
        self.assertLessEqual(norm_vec.max().item(), 1.0)

    def test_combine_all_scores_baseline(self):
        # Spat weight 0.0 recovers exact baseline SS-TPT
        suit = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stab = torch.tensor([4.0, 3.0, 2.0, 1.0])
        combined, details = combine_all_scores(
            suit, stab, spat_score=None, alpha=0.4, spat_weight=0.0
        )
        suit_n = min_max_normalize(suit)
        stab_n = min_max_normalize(stab)
        expected = 0.4 * suit_n + 0.6 * stab_n
        self.assertTrue(torch.allclose(combined, expected))
        self.assertIsNone(details["spat_n"])

    def test_combine_all_scores_linear(self):
        suit = torch.tensor([0.0, 1.0])
        stab = torch.tensor([0.0, 1.0])
        spat = torch.tensor([1.0, 0.0])
        combined, details = combine_all_scores(
            suit, stab, spat_score=spat, alpha=0.5, spat_weight=0.2, combine_mode="linear"
        )
        # ss_base = [0.0, 1.0], spat_n = [1.0, 0.0]
        # combined = ss_base + 0.2 * spat_n = [0.2, 1.0]
        self.assertTrue(torch.allclose(combined, torch.tensor([0.2, 1.0])))

    def test_combine_all_scores_gated_prevents_zombies(self):
        # View 0 is corrupted (suitability and stability are 0)
        # Even with high accidental spatial overlap (spat=1.0), gated mode keeps it at 0.0
        suit = torch.tensor([0.0, 5.0])
        stab = torch.tensor([0.0, 5.0])
        spat = torch.tensor([10.0, 5.0])  # View 0 has accidental spatial overlap
        combined, details = combine_all_scores(
            suit, stab, spat_score=spat, alpha=0.4, spat_weight=0.1, combine_mode="gated"
        )
        self.assertAlmostEqual(combined[0].item(), 0.0)
        self.assertGreater(combined[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()


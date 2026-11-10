"""Cross-view spatial relevance consistency scoring and score combination.

Provides:
- TinyAugmenterWithTransform: geometric weak augmentation tracking affine parameters.
- compute_pearson_spatial_similarity: zero-mean centered correlation on cropped heatmaps.
- combine_all_scores: unified scoring combining suitability, stability, and spatial consistency.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

try:
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = 2  # PIL bilinear interpolation fallback


class AddGaussianNoise(nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.02):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x):
        sigma = torch.empty(1, device=x.device).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0, 1)


class TinyAugmenterWithTransform(torch.nn.Module):
    """Weak augmentation module that explicitly tracks sampled affine parameters."""

    def __init__(
        self,
        rotation=5.0,
        translate=0.04,
        scale_var=0.08,
        brightness=0.08,
        contrast=0.08,
        saturation=0.08,
        hue=0.02,
        blur_sigma=(0.5, 1.2),
        noise_sigma=0.02,
    ):
        super().__init__()
        self.rotation = rotation
        self.translate = translate
        self.scale_var = scale_var

        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.blur_sigma = blur_sigma
        self.noise_sigma = noise_sigma

    def forward(self, x):
        B, C, H, W = x.shape
        angle = float(torch.empty(1).uniform_(-self.rotation, self.rotation).item())
        max_dx = float(self.translate * W)
        max_dy = float(self.translate * H)
        tx = float(torch.empty(1).uniform_(-max_dx, max_dx).item())
        ty = float(torch.empty(1).uniform_(-max_dy, max_dy).item())
        scale = float(torch.empty(1).uniform_(1.0 - self.scale_var, 1.0 + self.scale_var).item())

        # Apply affine transformation
        x_aug = TF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=BILINEAR,
        )

        # Photometric jitter
        if random.random() < 0.5:
            x_aug = self.color_jitter(x_aug)
        if random.random() < 0.5:
            sigma = float(torch.empty(1).uniform_(self.blur_sigma[0], self.blur_sigma[1]).item())
            x_aug = TF.gaussian_blur(x_aug, kernel_size=[5, 5], sigma=[sigma, sigma])
        if random.random() < 0.5:
            noise = torch.randn_like(x_aug) * (random.random() * self.noise_sigma)
            x_aug = (x_aug + noise).clamp(0, 1)

        transform_params = {
            "angle": angle,
            "translate": [tx, ty],
            "scale": scale,
            "shear": [0.0, 0.0],
        }
        return x_aug, transform_params

    def __call__(self, x):
        return self.forward(x)

    def align_spatial_map(self, spatial_map, transform_params):
        """Align a base spatial map with a weakly augmented view using recorded affine parameters."""
        return TF.affine(
            spatial_map,
            angle=transform_params["angle"],
            translate=transform_params["translate"],
            scale=transform_params["scale"],
            shear=transform_params["shear"],
            interpolation=BILINEAR,
        )


def compute_spatial_cosine_similarity(aligned_map, target_map, eps=1e-8):
    """Direct spatial cosine similarity on aligned attention maps (used in streamlined sstpt.py).
    
    aligned_map: [B, 1, H, W]
    target_map:  [B, 1, H, W]
    Returns: [B] similarity in [-1, 1]
    """
    return F.cosine_similarity(aligned_map.flatten(1), target_map.flatten(1), dim=1, eps=eps)


def compute_pearson_spatial_similarity(aligned_map, target_map, crop_margin=0, eps=1e-8):
    """Zero-mean centered Pearson spatial correlation with optional boundary cropping."""
    if crop_margin > 0 and aligned_map.shape[-2] > 2 * crop_margin and aligned_map.shape[-1] > 2 * crop_margin:
        aligned_crop = aligned_map[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
        target_crop = target_map[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
    else:
        aligned_crop = aligned_map
        target_crop = target_map

    u = aligned_crop.flatten(1)
    v = target_crop.flatten(1)

    u_c = u - u.mean(dim=1, keepdim=True)
    v_c = v - v.mean(dim=1, keepdim=True)

    sim = F.cosine_similarity(u_c, v_c, dim=1, eps=eps)
    return sim


def min_max_normalize(score, eps=1e-8):
    """Min-max normalize a 1-D score tensor to [0, 1]."""
    s_min = score.min()
    s_max = score.max()
    denom = s_max - s_min
    if denom < eps:
        return torch.zeros_like(score)
    return (score - s_min) / (denom + eps)


def combine_all_scores(
    suit_score,
    stab_score,
    spat_score=None,
    alpha=0.4,
    spat_weight=0.0,
    combine_mode="gated",
    eps=1e-8,
):
    """Unified score combination supporting Baseline, Task 1, Task 2, and Combined.

    Args:
        suit_score: 1-D tensor of suitability scores.
        stab_score: 1-D tensor of stability scores.
        spat_score: Optional 1-D tensor of spatial consistency scores.
        alpha: Scalar trade-off between suitability and stability (fixed or instance-adaptive alpha(x)).
        spat_weight: Weight gamma for spatial consistency (0.0 disables spatial).
        combine_mode: 'gated' (multiplicative reliability gate), 'linear' (S_base + gamma * spat),
                      or 'linear_norm' ((1 - gamma) * S_base + gamma * spat).
        eps: Small epsilon for numerical stability.

    Returns:
        combined_score: 1-D tensor of final quality scores for view weighting.
        detail_dict: Dictionary containing normalized components (suit_n, stab_n, spat_n, ss_base).
    """
    suit_n = min_max_normalize(suit_score, eps)
    stab_n = min_max_normalize(stab_score, eps)

    # Ensure alpha is a scalar tensor on the correct device
    if not torch.is_tensor(alpha):
        alpha_t = suit_score.new_tensor(float(alpha))
    else:
        alpha_t = alpha.to(device=suit_score.device, dtype=suit_score.dtype)

    ss_base = alpha_t * suit_n + (1.0 - alpha_t) * stab_n

    if spat_score is not None and spat_weight > 0.0:
        spat_n = min_max_normalize(spat_score, eps)
        if combine_mode == "linear":
            # Direct additive combination: S_base + gamma * C_spatial
            combined = ss_base + spat_weight * spat_n
        elif combine_mode == "linear_norm":
            # Normalized convex combination: (1 - gamma) * S_base + gamma * C_spatial
            combined = (1.0 - spat_weight) * ss_base + spat_weight * spat_n
        elif combine_mode == "gated":
            # Multiplicative reliability gate: S_base * ((1 - gamma) + gamma * C_spatial)
            combined = ss_base * ((1.0 - spat_weight) + spat_weight * spat_n)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
    else:
        spat_n = None
        combined = ss_base

    details = {
        "suit_n": suit_n,
        "stab_n": stab_n,
        "spat_n": spat_n,
        "ss_base": ss_base,
        "alpha": alpha_t,
    }
    return combined, details


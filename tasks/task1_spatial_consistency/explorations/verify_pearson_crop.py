#!/usr/bin/env python3
"""
Direct Empirical Verification of Pearson Zero-Mean Centering & 16px Boundary Cropping
Evaluates whether Pearson vs Raw Cosine and 16px Crop vs No Crop make any difference
in accuracy, view weights, and prediction flips on identical sample views.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from clip.custom_clip import get_coop
from data.cls_to_names import caltech101_classes
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
from data.datautils import build_dataset, AugMixAugmenter
from sstpt import TinyAugmenterWithTransform, js_divergence
import torchattacks

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, required=True, choices=["clean", "adv", "r"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--samples", type=int, default=500)
parser.add_argument("--data_root", type=str, default="/home/abhishek/workspace/SS-TPT/data")
args = parser.parse_args()

device = f"cuda:{args.gpu}"
torch.cuda.set_device(args.gpu)

# Dataset configuration
base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
preprocess = transforms.Compose([transforms.ToTensor()])

if args.domain == "clean":
    dset_name = "Caltech101"
    classnames = caltech101_classes
    eps, steps, alpha = 0.0, 0, 0.0
    augmix = True
elif args.domain == "adv":
    dset_name = "Caltech101"
    classnames = caltech101_classes
    eps, steps, alpha = 1.0, 7, 0.25
    augmix = True
elif args.domain == "r":
    dset_name = "R"
    label_mask = imagenet_r_mask
    classnames = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]
    eps, steps, alpha = 0.0, 0, 0.0
    augmix = False

data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=augmix)
val_dataset = build_dataset(dset_name, data_transform, args.data_root, mode="test")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = get_coop("RN50", classnames, str(device), 4, "a_photo_of_a", learned_cls=False)
model.eval()
initial_ctx = model.prompt_learner.ctx.clone().detach()

atk = torchattacks.PGD(model, eps=eps/255.0, alpha=alpha/255.0, steps=steps) if eps > 0 else None

with torch.no_grad():
    text_features = F.normalize(model.get_text_features(), dim=1)
    logit_scale = model.logit_scale.exp()

tiny_aug = TinyAugmenterWithTransform()

def entropy_avg(outputs):
    return (-(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()

def self_consistency_loss_loo(outputs, weight):
    eps_val = 1e-8
    probs = outputs.softmax(1).clamp_min(eps_val)
    weight = weight.unsqueeze(1)
    weighted_probs = weight * probs
    weighted_probs_sum = torch.sum(weighted_probs, dim=0, keepdim=True)
    loo_num = weighted_probs_sum - weighted_probs
    loo_den = (weight.sum(dim=0, keepdim=True) - weight).clamp_min(eps_val)
    ref_probs = (loo_num / loo_den).detach()
    return F.kl_div(probs.log(), ref_probs, reduction='batchmean', log_target=False)

configs = [
    {"name": "1_Baseline_NoSpatial",     "spat": "none",    "crop": 0,  "w": 0.00},
    {"name": "2_Pearson_16pxCrop",       "spat": "pearson", "crop": 16, "w": 0.10}, # Default task1 solution
    {"name": "3_RawCosine_16pxCrop",     "spat": "cosine",  "crop": 16, "w": 0.10}, # Raw cosine + crop
    {"name": "4_Pearson_NoCrop",         "spat": "pearson", "crop": 0,  "w": 0.10}, # Pearson + no crop
    {"name": "5_RawCosine_NoCrop",       "spat": "cosine",  "crop": 0,  "w": 0.10}, # Raw cosine + no crop
]

counts = {c["name"]: 0 for c in configs}
preds_history = {c["name"]: [] for c in configs}

# Metric stat trackers
stats = {
    "pearson_crop16": [],
    "cosine_crop16": [],
    "pearson_nocrop": [],
    "cosine_nocrop": [],
}

total = 0
t0 = time.time()
print(f"[{args.domain.upper()}] Starting Verification on {args.samples} samples on GPU {args.gpu}...")
print(f"Comparing 5 configs: Baseline vs (Pearson vs Raw Cosine) x (16px Crop vs No Crop)...", flush=True)

for i, (images, target) in enumerate(val_loader):
    if i >= args.samples:
        break
    target = target.to(device, non_blocking=True)

    if eps > 0.0:
        image = images[0].to(device, non_blocking=True)
        adv_image = atk(image, target)
        img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
        images = data_transform(img_adv)
        images = [_.unsqueeze(0) for _ in images]

    images = torch.cat([img.to(device, non_blocking=True) for img in images], dim=0)
    B = images.shape[0]

    # 1. Base attention map & features
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()
    model.image_encoder.attnpool.return_attn = False
    base_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    base_attn_224 = F.interpolate(base_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()

    # 2. Suitability (density)
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(B, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
    suit_n = suit_n.detach()

    # 3. Weak augmentation (K=1) & Prediction Stability
    aug_views, t_params = tiny_aug(images)
    with torch.no_grad():
        q_logits = model(aug_views)
    d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
    stab_score = 1.0 / (d + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
    stab_n = stab_n.detach()

    ss_base = (0.40 * suit_n + 0.60 * stab_n).detach()

    # 4. Weak Attention Map & Affine Realignment
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_attn_224 = F.interpolate(weak_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()
    aligned_attn_224 = tiny_aug.align_spatial_map(base_attn_224, t_params)

    # 5. Extract the 4 Spatial Score Variants:
    # A: 16px Cropped
    crop_aligned = aligned_attn_224[:, :, 16:-16, 16:-16].flatten(1)
    crop_weak = weak_attn_224[:, :, 16:-16, 16:-16].flatten(1)
    # Cosine crop
    cos_crop = F.cosine_similarity(crop_aligned, crop_weak, dim=1, eps=1e-8).detach()
    # Pearson crop
    u_c = crop_aligned - crop_aligned.mean(dim=1, keepdim=True)
    v_c = crop_weak - crop_weak.mean(dim=1, keepdim=True)
    pearson_crop = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8).detach()

    # B: No Crop (Full 224x224)
    nocrop_aligned = aligned_attn_224.flatten(1)
    nocrop_weak = weak_attn_224.flatten(1)
    # Cosine nocrop
    cos_nocrop = F.cosine_similarity(nocrop_aligned, nocrop_weak, dim=1, eps=1e-8).detach()
    # Pearson nocrop
    u_nc = nocrop_aligned - nocrop_aligned.mean(dim=1, keepdim=True)
    v_nc = nocrop_weak - nocrop_weak.mean(dim=1, keepdim=True)
    pearson_nocrop = F.cosine_similarity(u_nc, v_nc, dim=1, eps=1e-8).detach()

    # Track metric statistics
    stats["pearson_crop16"].append(pearson_crop.cpu().numpy())
    stats["cosine_crop16"].append(cos_crop.cpu().numpy())
    stats["pearson_nocrop"].append(pearson_nocrop.cpu().numpy())
    stats["cosine_nocrop"].append(cos_nocrop.cpu().numpy())

    # Min-max normalization for each variant
    def norm_s(s):
        return ((s - s.min()) / (s.max() - s.min() + 1e-8)).detach()

    spat_dict = {
        "none": None,
        "pearson_crop16": norm_s(pearson_crop),
        "cosine_crop16":  norm_s(cos_crop),
        "pearson_nocrop": norm_s(pearson_nocrop),
        "cosine_nocrop":  norm_s(cos_nocrop),
    }

    # 6. Evaluate all 5 configs on the EXACT same sample
    tau = 0.25
    for cfg in configs:
        cname = cfg["name"]
        stype = cfg["spat"]
        crop = cfg["crop"]
        w = cfg["w"]

        if stype == "none":
            score_vec = ss_base
        else:
            k_key = f"{stype}_crop{crop}" if crop > 0 else f"{stype}_nocrop"
            spat_val = spat_dict[k_key]
            # Multiplicative Gating (w=0.10)
            score_vec = ss_base * ((1.0 - w) + w * spat_val)
        score_vec = score_vec.detach()

        # Reset prompt context
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)

        # 1-step prompt tuning on top-3 lowest entropy views
        out = model(images)
        ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)
        sel = torch.topk(ent, 3, largest=False).indices

        weight_cons = F.softmax(score_vec[sel] / tau, dim=-1)
        ent_loss = entropy_avg(out[sel])
        kl_loss = self_consistency_loss_loo(out[sel], weight_cons)
        loss = ent_loss + 1.0 * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Weighted prediction
        with torch.no_grad():
            weight_pred = F.softmax(score_vec / tau, dim=-1)
            pred = (weight_pred.unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            if pred == target:
                counts[cname] += 1
            preds_history[cname].append(pred.item())

    total += 1
    if (i + 1) % 50 == 0 or (i + 1) == args.samples:
        elapsed = time.time() - t0
        b_acc = counts["1_Baseline_NoSpatial"] / total * 100.0
        p_c_acc = counts["2_Pearson_16pxCrop"] / total * 100.0
        cos_c_acc = counts["3_RawCosine_16pxCrop"] / total * 100.0
        p_nc_acc = counts["4_Pearson_NoCrop"] / total * 100.0
        print(f"[{args.domain.upper()}] Sample {i+1}/{args.samples} ({elapsed:.1f}s) | Base: {b_acc:.1f}% | Pearson+Crop: {p_c_acc:.1f}% | Cos+Crop: {cos_c_acc:.1f}% | Pearson+NoCrop: {p_nc_acc:.1f}%", flush=True)

# Final summary & Flips analysis
elapsed = time.time() - t0
base_acc = counts["1_Baseline_NoSpatial"] / total * 100.0

print(f"\n=======================================================")
print(f"[{args.domain.upper()}] Final Results on {total} samples ({elapsed:.1f}s):")
print(f"=======================================================")
results = {}
for cfg in configs:
    name = cfg["name"]
    acc = counts[name] / total * 100.0
    diff = acc - base_acc
    results[name] = {"acc": acc, "diff": diff, "correct": counts[name], "total": total}
    print(f"  {name:<28} : {acc:6.2f}% ({diff:+5.2f}%)  [{counts[name]}/{total}]")

# Flip analysis:
p_preds = np.array(preds_history["2_Pearson_16pxCrop"])
cos_preds = np.array(preds_history["3_RawCosine_16pxCrop"])
nocrop_preds = np.array(preds_history["4_Pearson_NoCrop"])
base_preds = np.array(preds_history["1_Baseline_NoSpatial"])

p_vs_cos_flips = np.sum(p_preds != cos_preds)
crop_vs_nocrop_flips = np.sum(p_preds != nocrop_preds)
p_vs_base_flips = np.sum(p_preds != base_preds)

print(f"\n--- Prediction Disagreement / Flip Analysis ---")
print(f"  Total samples where Pearson vs Raw Cosine disagreed : {p_vs_cos_flips} / {total} ({p_vs_cos_flips/total*100:.2f}%)")
print(f"  Total samples where 16px Crop vs No Crop disagreed : {crop_vs_nocrop_flips} / {total} ({crop_vs_nocrop_flips/total*100:.2f}%)")
print(f"  Total samples where Pearson+Crop vs Baseline disagreed : {p_vs_base_flips} / {total} ({p_vs_base_flips/total*100:.2f}%)")

# Metric values distribution
all_p_c = np.concatenate(stats["pearson_crop16"])
all_cos_c = np.concatenate(stats["cosine_crop16"])
all_p_nc = np.concatenate(stats["pearson_nocrop"])
all_cos_nc = np.concatenate(stats["cosine_nocrop"])

metric_summary = {
    "pearson_crop16": {"mean": float(np.mean(all_p_c)), "std": float(np.std(all_p_c)), "min": float(np.min(all_p_c)), "max": float(np.max(all_p_c))},
    "cosine_crop16":  {"mean": float(np.mean(all_cos_c)), "std": float(np.std(all_cos_c)), "min": float(np.min(all_cos_c)), "max": float(np.max(all_cos_c))},
    "pearson_nocrop": {"mean": float(np.mean(all_p_nc)), "std": float(np.std(all_p_nc)), "min": float(np.min(all_p_nc)), "max": float(np.max(all_p_nc))},
    "cosine_nocrop":  {"mean": float(np.mean(all_cos_nc)), "std": float(np.std(all_cos_nc)), "min": float(np.min(all_cos_nc)), "max": float(np.max(all_cos_nc))},
}

print(f"\n--- Metric Value Distribution (Raw Values across all views) ---")
print(f"  Raw Cosine (16px Crop) : Mean = {metric_summary['cosine_crop16']['mean']:.4f}, Std = {metric_summary['cosine_crop16']['std']:.4f}, Min = {metric_summary['cosine_crop16']['min']:.4f}, Max = {metric_summary['cosine_crop16']['max']:.4f}")
print(f"  Pearson    (16px Crop) : Mean = {metric_summary['pearson_crop16']['mean']:.4f}, Std = {metric_summary['pearson_crop16']['std']:.4f}, Min = {metric_summary['pearson_crop16']['min']:.4f}, Max = {metric_summary['pearson_crop16']['max']:.4f}")
print(f"  Raw Cosine (No Crop)   : Mean = {metric_summary['cosine_nocrop']['mean']:.4f}, Std = {metric_summary['cosine_nocrop']['std']:.4f}, Min = {metric_summary['cosine_nocrop']['min']:.4f}, Max = {metric_summary['cosine_nocrop']['max']:.4f}")
print(f"  Pearson    (No Crop)   : Mean = {metric_summary['pearson_nocrop']['mean']:.4f}, Std = {metric_summary['pearson_nocrop']['std']:.4f}, Min = {metric_summary['pearson_nocrop']['min']:.4f}, Max = {metric_summary['pearson_nocrop']['max']:.4f}")

out_data = {
    "results": results,
    "flips": {
        "pearson_vs_cosine": int(p_vs_cos_flips),
        "crop_vs_nocrop": int(crop_vs_nocrop_flips),
        "pearson_vs_base": int(p_vs_base_flips),
    },
    "metric_summary": metric_summary
}

with open(f"verify_pearson_crop_{args.domain}.json", "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nSaved full verification data to verify_pearson_crop_{args.domain}.json\n")


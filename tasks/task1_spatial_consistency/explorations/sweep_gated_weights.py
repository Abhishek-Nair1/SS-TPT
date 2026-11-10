#!/usr/bin/env python3
"""
Systematic Evaluation & Sweep of the Weighting Parameter in Multiplicative Gating
Formula: Score = SS_Base * ((1 - w) + w * Spatial_Norm)
Evaluates w in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] on identical views
using official Leave-One-Out loss and direct spatial cosine similarity.
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

# Multiplicative gating weights to test
test_weights = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
configs = [
    {"name": f"gated_w{w:.2f}", "w": w} for w in test_weights
]

counts = {c["name"]: 0 for c in configs}
preds_history = {c["name"]: [] for c in configs}

# Track weight shift metrics
weight_shifts = {c["name"]: [] for c in configs if c["w"] > 0.0}

total = 0
t0 = time.time()
print(f"[{args.domain.upper()}] Starting Multiplicative Gating Weight Sweep on {args.samples} samples on GPU {args.gpu}...")
print(f"Testing weights: {test_weights} with official Leave-One-Out loss...", flush=True)

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

    # 1. Base forward pass & attention map
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

    # 4. Weak attention map & Direct Cosine Alignment (No crop, no mean subtraction)
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_attn_224 = F.interpolate(weak_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()
    aligned_attn_224 = tiny_aug.align_spatial_map(base_attn_224, t_params)

    # Direct spatial cosine similarity
    spat_raw = F.cosine_similarity(aligned_attn_224.flatten(1), weak_attn_224.flatten(1), dim=1, eps=1e-8).detach()
    spat_n = ((spat_raw - spat_raw.min()) / (spat_raw.max() - spat_raw.min() + 1e-8)).detach()

    # Base weight distribution for reference (w=0.0)
    tau = 0.25
    base_weight = F.softmax(ss_base / tau, dim=-1)

    # 5. Evaluate each gating weight configuration
    for cfg in configs:
        cname = cfg["name"]
        w = cfg["w"]

        if w == 0.0:
            score_vec = ss_base
        else:
            # Multiplicative Gating
            score_vec = ss_base * ((1.0 - w) + w * spat_n)
            
            # Log L1 shift in view weights compared to baseline
            cur_weight = F.softmax(score_vec / tau, dim=-1)
            l1_shift = torch.sum(torch.abs(cur_weight - base_weight)).item()
            weight_shifts[cname].append(l1_shift)

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

        # Weighted test-time prediction
        with torch.no_grad():
            weight_pred = F.softmax(score_vec / tau, dim=-1)
            pred = (weight_pred.unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            if pred == target:
                counts[cname] += 1
            preds_history[cname].append(pred.item())

    total += 1
    if (i + 1) % 50 == 0 or (i + 1) == args.samples:
        elapsed = time.time() - t0
        base_acc = counts["gated_w0.00"] / total * 100.0
        w10_acc = counts["gated_w0.10"] / total * 100.0
        w20_acc = counts["gated_w0.20"] / total * 100.0
        print(f"[{args.domain.upper()}] Sample {i+1}/{args.samples} ({elapsed:.1f}s) | w=0.00: {base_acc:.1f}% | w=0.10: {w10_acc:.1f}% | w=0.20: {w20_acc:.1f}%", flush=True)

# Final summary
elapsed = time.time() - t0
base_acc = counts["gated_w0.00"] / total * 100.0

print(f"\n=======================================================")
print(f"[{args.domain.upper()}] Final Weight Sweep Results on {total} samples ({elapsed:.1f}s):")
print(f"=======================================================")

results = {}
for cfg in configs:
    name = cfg["name"]
    w = cfg["w"]
    acc = counts[name] / total * 100.0
    diff = acc - base_acc
    avg_l1 = float(np.mean(weight_shifts[name])) if w > 0.0 else 0.0
    results[name] = {
        "weight": w,
        "acc": acc,
        "diff": diff,
        "correct": counts[name],
        "total": total,
        "avg_l1_weight_shift": avg_l1
    }
    print(f"  {name:<14} (w={w:.2f}) : {acc:6.2f}% ({diff:+5.2f}%)  [{counts[name]}/{total}]  Avg L1 Weight Shift = {avg_l1:.4f}")

out_file = f"gated_weight_sweep_{args.domain}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {out_file}\n")


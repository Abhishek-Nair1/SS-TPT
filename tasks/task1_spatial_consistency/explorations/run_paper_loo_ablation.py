import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import argparse

from torchvision import transforms
import torchvision.transforms.functional as TF
from clip.custom_clip import get_coop
from data.cls_to_names import caltech101_classes
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
from data.datautils import build_dataset, AugMixAugmenter
from sstpt import TinyAugmenterWithTransform, js_divergence, self_consistency_loss_with_rs_view
import torchattacks

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, required=True, choices=['clean', 'adv', 'r'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--samples', type=int, default=2000)
parser.add_argument('--data_root', type=str, default='/home/abhishek/workspace/SS-TPT/data')
args = parser.parse_args()

device = f"cuda:{args.gpu}"
torch.cuda.set_device(args.gpu)

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
preprocess = transforms.Compose([transforms.ToTensor()])

if args.domain == 'clean':
    dset_name = 'Caltech101'
    classnames = caltech101_classes
    eps, steps, alpha = 0.0, 0, 0.0
    augmix = True
elif args.domain == 'adv':
    dset_name = 'Caltech101'
    classnames = caltech101_classes
    eps, steps, alpha = 1.0, 7, 0.25
    augmix = True
elif args.domain == 'r':
    dset_name = 'R'
    label_mask = imagenet_r_mask
    classnames = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]
    eps, steps, alpha = 0.0, 0, 0.0
    augmix = False

data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=augmix)
val_dataset = build_dataset(dset_name, data_transform, args.data_root, mode='test')
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

# 5 Core Configurations
configs = [
    # 1. Official Paper Baseline with Leave-One-Out (LOO) Consistency Loss
    {"name": "1_Baseline_Paper_LOO", "spat": "none", "combine": "none", "w": 0.0, "select": "entropy"},
    
    # 2. Gated S3-TPT with Min-Max Normalized Pearson (Our previous winning Method 4)
    {"name": "2_Gated_MinMax_Pearson", "spat": "minmax", "combine": "gated", "w": 0.10, "select": "entropy"},
    
    # 3. Gated S3-TPT with Absolute Clamped Pearson (NO MIN-MAX! The New Hypothesis)
    {"name": "3_Gated_Absolute_Pearson", "spat": "absolute", "combine": "gated", "w": 0.10, "select": "entropy"},
    
    # 4. Linear S3-TPT with Min-Max Normalized Pearson
    {"name": "4_Linear_MinMax_Pearson", "spat": "minmax", "combine": "linear", "w": 0.20, "select": "entropy"},
    
    # 5. Linear S3-TPT with Absolute Clamped Pearson (NO MIN-MAX)
    {"name": "5_Linear_Absolute_Pearson", "spat": "absolute", "combine": "linear", "w": 0.20, "select": "entropy"},
]

counts = {c["name"]: 0 for c in configs}
total = 0
t0 = time.time()

print(f"[{args.domain.upper()}] Starting Paper LOO Evaluation on {args.samples} samples on GPU {args.gpu}...")
print(f"[{args.domain.upper()}] Evaluating {len(configs)} configurations per sample using official Leave-One-Out loss...", flush=True)

for i, (images, target) in enumerate(val_loader):
    if i >= args.samples:
        break
    target = target.to(device, non_blocking=True)
    
    # Generate PGD attack ONCE per sample
    if eps > 0.0:
        image = images[0].to(device, non_blocking=True)
        adv_image = atk(image, target)
        img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
        images = data_transform(img_adv)
        images = [_.unsqueeze(0) for _ in images]

    images = torch.cat([img.to(device, non_blocking=True) for img in images], dim=0)
    B = images.shape[0]

    # 1. Base feature pass & AttentionPool map (Native forward hook, zero backprop)
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()
    model.image_encoder.attnpool.return_attn = False
    base_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    base_attn_224 = F.interpolate(base_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()

    # 2. Suitability (Exact Density Mode from sstpt.py)
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(B, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
    suit_n = suit_n.detach()

    # 3. Weak augmentation (K=1) & Prediction Stability (JS Divergence)
    aug_views, t_params = tiny_aug(images)
    with torch.no_grad():
        q_logits = model(aug_views)
    d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
    stab_score = 1.0 / (d + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
    stab_n = stab_n.detach()

    ss_base = (0.40 * suit_n + 0.60 * stab_n).detach()

    # 4. Weak Spatial Attention Map & Affine Realignment
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_attn_224 = F.interpolate(weak_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()
    aligned_attn_224 = tiny_aug.align_spatial_map(base_attn_224, t_params)

    # 5. 16px Boundary Cropping & Pearson Zero-Mean Centering
    crop_base = aligned_attn_224[:, :, 16:-16, 16:-16].flatten(1)
    crop_weak = weak_attn_224[:, :, 16:-16, 16:-16].flatten(1)

    u_c = crop_base - crop_base.mean(dim=1, keepdim=True)
    v_c = crop_weak - crop_weak.mean(dim=1, keepdim=True)
    r_pearson = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8).detach()

    # Variant A: Min-Max Normalized Pearson (Relative spread)
    r_minmax = ((r_pearson - r_pearson.min()) / (r_pearson.max() - r_pearson.min() + 1e-8)).detach()

    # Variant B: Absolute Clamped Pearson (NO Min-Max, preserves absolute physical meaning!)
    r_absolute = torch.clamp(r_pearson, min=0.0, max=1.0).detach()

    # 6. Evaluate each configuration with official Leave-One-Out consistency loss
    tau = 0.25
    for cfg in configs:
        cname = cfg["name"]
        spat_type = cfg["spat"]
        comb_type = cfg["combine"]
        w = cfg["w"]

        if comb_type == "none":
            score_vec = ss_base
        elif comb_type == "linear":
            spat_val = r_minmax if spat_type == "minmax" else r_absolute
            score_vec = (1.0 - w) * ss_base + w * spat_val
        elif comb_type == "gated":
            spat_val = r_minmax if spat_type == "minmax" else r_absolute
            score_vec = ss_base * ((1.0 - w) + w * spat_val)
        score_vec = score_vec.detach()

        # Reset model prompt to initial context
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)

        # Prompt tuning (1 step) with pure entropy selection
        out = model(images)
        ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)
        sel = torch.topk(ent, 3, largest=False).indices

        weight_cons = F.softmax(score_vec[sel] / tau, dim=-1)
        
        # EXACT Leave-One-Out (LOO) self-consistency loss from sstpt.py
        ent_loss = entropy_avg(out[sel])
        kl_loss = self_consistency_loss_with_rs_view(out[sel], weight_cons, ref_mode="weighted")
        loss = ent_loss + 1.0 * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Final prediction with temperature-weighted consensus
        with torch.no_grad():
            weight_pred = F.softmax(score_vec / tau, dim=-1)
            pred = (weight_pred.unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            if pred == target:
                counts[cname] += 1

    total += 1
    if (i + 1) % 25 == 0 or (i + 1) == args.samples:
        elapsed = time.time() - t0
        base_acc = counts["1_Baseline_Paper_LOO"] / total * 100.0
        gated_abs_acc = counts["3_Gated_Absolute_Pearson"] / total * 100.0
        print(f"[{args.domain.upper()}] Sample {i+1}/{args.samples} ({elapsed:.1f}s) | Base (LOO): {base_acc:.1f}% | Gated Abs: {gated_abs_acc:.1f}%", flush=True)

        if (i + 1) % 100 == 0:
            print(f"\n--- [{args.domain.upper()}] Snapshot at Sample {i+1}/{args.samples} ({elapsed:.1f}s) ---", flush=True)
            for cfg in configs:
                c_name = cfg["name"]
                cur_acc = counts[c_name] / total * 100.0
                cur_diff = cur_acc - base_acc
                print(f"  {c_name:<30}: {cur_acc:6.2f}% ({cur_diff:+5.2f}%)", flush=True)
            print("-" * 60 + "\n", flush=True)

# Final summary
elapsed = time.time() - t0
base_acc = counts["1_Baseline_Paper_LOO"] / total * 100.0
print(f"\n[{args.domain.upper()}] Final Results on {total} samples ({elapsed:.1f}s):")
print("=" * 65)
results = {}
for cfg in configs:
    name = cfg["name"]
    acc = counts[name] / total * 100.0
    diff = acc - base_acc
    results[name] = {"acc": acc, "diff": diff, "correct": counts[name], "total": total}
    print(f"{name:<30} : {acc:6.2f}% ({diff:+5.2f}%)")

out_file = f"paper_loo_results_{args.domain}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {out_file}\n")


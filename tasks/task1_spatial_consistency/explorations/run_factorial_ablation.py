import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
import argparse

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
parser.add_argument('--domain', type=str, required=True, choices=['clean', 'adv', 'r'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--data_root', type=str, default='/home/abhishek/workspace/SS-TPT/data')
args = parser.parse_args()

if args.domain == 'r' and args.gpu == 0:
    if os.path.exists('ablation_gpu2_r.log') or os.path.exists('factorial_ablation_r.json'):
        print("[R] ImageNet-R is already running or completed on GPU 2! Skipping duplicate run on GPU 0.")
        sys.exit(0)

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

def self_consistency_loss(outputs, weight):
    probs = outputs.softmax(1).clamp_min(1e-8)
    p_bar = (weight.unsqueeze(1) * probs).sum(0, keepdim=True)
    return F.kl_div(outputs.log_softmax(1), p_bar.repeat(outputs.size(0), 1), reduction='batchmean')

configs = [
    # 1. Baseline
    {"name": "1_Baseline_SS-TPT", "spat": "none", "combine": "none", "w": 0.0, "select": "entropy"},
    
    # 2. The 2x2 Factorial Grid (Extractor x Selection Mode) with Linear addition
    {"name": "2_GradCAM_Score_Linear", "spat": "cam_raw", "combine": "linear", "w": 0.20, "select": "score"},
    {"name": "3_GradCAM_Entropy_Linear", "spat": "cam_raw", "combine": "linear", "w": 0.20, "select": "entropy"},
    {"name": "4_Attn_Score_Linear", "spat": "attn_raw", "combine": "linear", "w": 0.20, "select": "score"},
    {"name": "5_Attn_Entropy_Linear", "spat": "attn_raw", "combine": "linear", "w": 0.20, "select": "entropy"},
    
    # 3. Effect of Pearson Correlation (Removing Positive DC bias)
    {"name": "6_GradCAM_Entropy_Pearson_Linear", "spat": "cam_pearson", "combine": "linear", "w": 0.20, "select": "entropy"},
    {"name": "7_Attn_Entropy_Pearson_Linear", "spat": "attn_pearson", "combine": "linear", "w": 0.20, "select": "entropy"},
    
    # 4. Multiplicative Reliability Gating (Method 4 Winning Innovation)
    {"name": "8_Attn_Entropy_Gated", "spat": "attn_pearson", "combine": "gated", "w": 0.10, "select": "entropy"},
    {"name": "9_GradCAM_Entropy_Gated", "spat": "cam_pearson", "combine": "gated", "w": 0.10, "select": "entropy"},
    {"name": "10_Attn_Score_Gated", "spat": "attn_pearson", "combine": "gated", "w": 0.10, "select": "score"},
    
    # 5. Boundary Crop Ablation (16px Crop vs No Crop)
    {"name": "11_Attn_Entropy_NoCrop_Gated", "spat": "attn_nocrop", "combine": "gated", "w": 0.10, "select": "entropy"},
]

counts = {c["name"]: 0 for c in configs}
total = 0
t0 = time.time()

print(f"[{args.domain.upper()}] Starting Factorial Ablation on {args.samples} samples on GPU {args.gpu}...")
print(f"[{args.domain.upper()}] Evaluating {len(configs)} configurations per sample...", flush=True)

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

    # 1. Base feature pass & AttentionPool map
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()
    model.image_encoder.attnpool.return_attn = False
    base_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    base_attn_224 = F.interpolate(base_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()

    # 2. Grad-CAM base map
    base_cam_224 = model.get_spatial_relevance_map(images, method='gradcam').detach()

    # 3. Suitability (k-NN feature density)
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(B, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)

    # 4. Weak augmentation (K=1)
    aug_views, t_params = tiny_aug(images)
    with torch.no_grad():
        q_logits = model(aug_views)
    d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
    stab_score = 1.0 / (d + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)

    ss_base = (0.40 * suit_n + 0.60 * stab_n).detach()

    # 5. Weak maps & Realignment
    # Attention weak map
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_attn_224 = F.interpolate(weak_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False).detach()
    aligned_attn_224 = tiny_aug.align_spatial_map(base_attn_224, t_params)

    # Grad-CAM weak map
    weak_cam_224 = model.get_spatial_relevance_map(aug_views, method='gradcam').detach()
    aligned_cam_224 = tiny_aug.align_spatial_map(base_cam_224, t_params)

    # 6. Compute spatial metrics
    # Crop 16px
    crop_attn_base = aligned_attn_224[:, :, 16:-16, 16:-16].flatten(1)
    crop_attn_weak = weak_attn_224[:, :, 16:-16, 16:-16].flatten(1)
    crop_cam_base = aligned_cam_224[:, :, 16:-16, 16:-16].flatten(1)
    crop_cam_weak = weak_cam_224[:, :, 16:-16, 16:-16].flatten(1)

    # Attention Raw Cosine
    sim_attn_raw = F.cosine_similarity(crop_attn_base, crop_attn_weak, dim=1, eps=1e-8)
    spat_attn_raw_n = (sim_attn_raw - sim_attn_raw.min()) / (sim_attn_raw.max() - sim_attn_raw.min() + 1e-8)

    # Attention Pearson (Zero-mean centered)
    u_attn_c = crop_attn_base - crop_attn_base.mean(dim=1, keepdim=True)
    v_attn_c = crop_attn_weak - crop_attn_weak.mean(dim=1, keepdim=True)
    sim_attn_pearson = F.cosine_similarity(u_attn_c, v_attn_c, dim=1, eps=1e-8)
    spat_attn_pearson_n = (sim_attn_pearson - sim_attn_pearson.min()) / (sim_attn_pearson.max() - sim_attn_pearson.min() + 1e-8)

    # Attention No Crop (0px) Pearson
    u_nocrop = aligned_attn_224.flatten(1)
    v_nocrop = weak_attn_224.flatten(1)
    u_nc_c = u_nocrop - u_nocrop.mean(dim=1, keepdim=True)
    v_nc_c = v_nocrop - v_nocrop.mean(dim=1, keepdim=True)
    sim_attn_nocrop = F.cosine_similarity(u_nc_c, v_nc_c, dim=1, eps=1e-8)
    spat_attn_nocrop_n = (sim_attn_nocrop - sim_attn_nocrop.min()) / (sim_attn_nocrop.max() - sim_attn_nocrop.min() + 1e-8)

    # GradCAM Raw Cosine
    sim_cam_raw = F.cosine_similarity(crop_cam_base, crop_cam_weak, dim=1, eps=1e-8)
    spat_cam_raw_n = (sim_cam_raw - sim_cam_raw.min()) / (sim_cam_raw.max() - sim_cam_raw.min() + 1e-8)

    # GradCAM Pearson (Zero-mean centered)
    u_cam_c = crop_cam_base - crop_cam_base.mean(dim=1, keepdim=True)
    v_cam_c = crop_cam_weak - crop_cam_weak.mean(dim=1, keepdim=True)
    sim_cam_pearson = F.cosine_similarity(u_cam_c, v_cam_c, dim=1, eps=1e-8)
    spat_cam_pearson_n = (sim_cam_pearson - sim_cam_pearson.min()) / (sim_cam_pearson.max() - sim_cam_pearson.min() + 1e-8)

    spatial_metrics = {
        "none": None,
        "attn_raw": spat_attn_raw_n,
        "attn_pearson": spat_attn_pearson_n,
        "attn_nocrop": spat_attn_nocrop_n,
        "cam_raw": spat_cam_raw_n,
        "cam_pearson": spat_cam_pearson_n,
    }

    # Evaluate each configuration
    tau = 0.25
    for cfg in configs:
        cname = cfg["name"]
        spat_type = cfg["spat"]
        comb_type = cfg["combine"]
        w = cfg["w"]
        sel_mode = cfg["select"]

        if comb_type == "none":
            score_vec = ss_base
        elif comb_type == "linear":
            score_vec = (1.0 - w) * ss_base + w * spatial_metrics[spat_type]
        elif comb_type == "gated":
            score_vec = ss_base * ((1.0 - w) + w * spatial_metrics[spat_type])

        # Test-Time Adaptation
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)

        out = model(images)
        if sel_mode == "score":
            sel = torch.topk(score_vec, 3, largest=True).indices
        else: # entropy
            ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)
            sel = torch.topk(ent, 3, largest=False).indices

        w_c = F.softmax(score_vec[sel] / tau, dim=-1)
        loss = entropy_avg(out[sel]) + 1.0 * self_consistency_loss(out[sel], w_c)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            weight_pred = F.softmax(score_vec / tau, dim=-1)
            pred = (weight_pred.unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            if pred == target:
                counts[cname] += 1

    total += 1
    if (i + 1) % 25 == 0 or (i + 1) == args.samples:
        elapsed = time.time() - t0
        base_acc = counts["1_Baseline_SS-TPT"] / total * 100.0
        gated_acc = counts["8_Attn_Entropy_Gated"] / total * 100.0
        print(f"[{args.domain.upper()}] Sample {i+1}/{args.samples} ({elapsed:.1f}s) | Base: {base_acc:.1f}% | Gated: {gated_acc:.1f}%", flush=True)
        if (i + 1) % 100 == 0:
            print(f"\n--- [{args.domain.upper()}] Interim Snapshot at Sample {i+1}/{args.samples} ({elapsed:.1f}s) ---", flush=True)
            for cfg in configs:
                c_name = cfg["name"]
                cur_acc = counts[c_name] / total * 100.0
                cur_diff = cur_acc - base_acc
                print(f"  {c_name:<35}: {cur_acc:6.2f}% ({cur_diff:+5.2f}%)", flush=True)
            print("-" * 65 + "\n", flush=True)

# Final summary
elapsed = time.time() - t0
base_acc = counts["1_Baseline_SS-TPT"] / total * 100.0
print(f"\n[{args.domain.upper()}] Final Results on {total} samples ({elapsed:.1f}s):")
print("-" * 75)
results = {}
for cfg in configs:
    name = cfg["name"]
    acc = counts[name] / total * 100.0
    diff = acc - base_acc
    results[name] = {"acc": acc, "diff": diff, "correct": counts[name], "total": total}
    print(f"{name:<35} : {acc:6.2f}% ({diff:+5.2f}%)")

out_file = f"factorial_ablation_{args.domain}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {out_file}\n")

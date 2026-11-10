import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import json
import argparse
import math

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
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--samples", type=int, default=150)
parser.add_argument("--data_root", type=str, default="/home/abhishek/workspace/SS-TPT/data")
args = parser.parse_args()

device = f"cuda:{args.gpu}"
torch.cuda.set_device(args.gpu)

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

def self_consistency_loss(outputs, weight):
    probs = outputs.softmax(1).clamp_min(1e-8)
    p_bar = (weight.unsqueeze(1) * probs).sum(0, keepdim=True)
    return F.kl_div(outputs.log_softmax(1), p_bar.repeat(outputs.size(0), 1), reduction="batchmean")

configs = [
    {"name": "baseline", "type": "baseline"},
    {"name": "pearson_lin_w0.08", "type": "pearson", "gate": "linear", "w": 0.08},
    {"name": "pearson_lin_w0.10", "type": "pearson", "gate": "linear", "w": 0.10},
    {"name": "pearson_lin_w0.12", "type": "pearson", "gate": "linear", "w": 0.12},
    {"name": "pearson_lin_w0.15", "type": "pearson", "gate": "linear", "w": 0.15},
    {"name": "pearson_lin_w0.20", "type": "pearson", "gate": "linear", "w": 0.20},
    {"name": "pearson_exp_w0.15", "type": "pearson", "gate": "exp", "w": 0.15},
    {"name": "pearson_exp_w0.20", "type": "pearson", "gate": "exp", "w": 0.20},
    {"name": "spat_js_w0.10", "type": "js", "gate": "linear", "w": 0.10},
    {"name": "spat_js_w0.15", "type": "js", "gate": "linear", "w": 0.15},
    {"name": "sal_pearson_w0.10", "type": "sal_pearson", "gate": "linear", "w": 0.10},
    {"name": "sal_pearson_w0.15", "type": "sal_pearson", "gate": "linear", "w": 0.15},
    {"name": "native7x7_w0.10", "type": "native7x7", "gate": "linear", "w": 0.10},
    {"name": "native7x7_w0.15", "type": "native7x7", "gate": "linear", "w": 0.15},
    {"name": "entropy_mod_w0.12", "type": "entropy_mod", "gate": "adaptive", "w": 0.12},
    {"name": "entropy_mod_w0.15", "type": "entropy_mod", "gate": "adaptive", "w": 0.15},
    {"name": "entropy_mod_w0.20", "type": "entropy_mod", "gate": "adaptive", "w": 0.20},
]

counts = {c["name"]: 0 for c in configs}
total = 0
t0 = time.time()

print(f"[{args.domain.upper()}] Starting Task 1 Exploration on {args.samples} samples on GPU {args.gpu}...")
print(f"[{args.domain.upper()}] Configurations under test: {len(configs)}")

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
    B = images.size(0)
    
    # 1. Base forward pass & attention map
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()
    model.image_encoder.attnpool.return_attn = False
    
    raw_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    base_map = F.interpolate(raw_attn_7x7, size=(224, 224), mode="bilinear", align_corners=False)
    
    # 2. Suitability score
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(16, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
    
    # 3. Weak augmentation (K=1)
    aug_views, t_params = tiny_aug(images)
    with torch.no_grad():
        q_logits = model(aug_views)
    d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
    stab_score = 1.0 / (d + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
    
    ss_base = 0.40 * suit_n + 0.60 * stab_n
    
    # 4. Weak spatial map & alignment
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_map = F.interpolate(weak_attn_7x7, size=(224, 224), mode="bilinear", align_corners=False)
    
    aligned_base = tiny_aug.align_spatial_map(base_map, t_params)
    
    # Border-cropped maps (16px margin)
    m_base = aligned_base[:, :, 16:-16, 16:-16]
    m_weak = weak_map[:, :, 16:-16, 16:-16]
    
    # 1: Aligned Pearson Correlation (Zero-Mean Centered)
    u = m_base.flatten(1)
    v = m_weak.flatten(1)
    u_c = u - u.mean(dim=1, keepdim=True)
    v_c = v - v.mean(dim=1, keepdim=True)
    spat_pearson = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8)
    spat_pearson_n = (spat_pearson - spat_pearson.min()) / (spat_pearson.max() - spat_pearson.min() + 1e-8)
    
    # 2: Spatial JS Divergence
    aligned_7x7 = F.adaptive_avg_pool2d(aligned_base, (7, 7)).view(B, -1)
    weak_7x7 = weak_attn_7x7.view(B, -1)
    p_spat = F.softmax(aligned_7x7 / 0.5, dim=-1)
    q_spat = F.softmax(weak_7x7 / 0.5, dim=-1)
    m_spat = 0.5 * (p_spat + q_spat)
    kl1 = (p_spat * (p_spat / m_spat.clamp_min(1e-8)).log()).sum(dim=-1)
    kl2 = (q_spat * (q_spat / m_spat.clamp_min(1e-8)).log()).sum(dim=-1)
    js_spat = 0.5 * (kl1 + kl2)
    spat_js = 1.0 / (js_spat + 1e-8)
    spat_js_n = (spat_js - spat_js.min()) / (spat_js.max() - spat_js.min() + 1e-8)
    
    # 3: Foreground Saliency-Weighted Pearson
    fg_mask = (m_base > m_base.mean(dim=(2, 3), keepdim=True)).float()
    u_fg = (m_base * fg_mask).flatten(1)
    v_fg = (m_weak * fg_mask).flatten(1)
    u_fg_c = u_fg - u_fg.mean(dim=1, keepdim=True)
    v_fg_c = v_fg - v_fg.mean(dim=1, keepdim=True)
    sal_pearson = F.cosine_similarity(u_fg_c, v_fg_c, dim=1, eps=1e-8)
    sal_pearson_n = (sal_pearson - sal_pearson.min()) / (sal_pearson.max() - sal_pearson.min() + 1e-8)
    
    # 4: Native 7x7 Patch Pearson
    u7 = aligned_7x7 - aligned_7x7.mean(dim=1, keepdim=True)
    v7 = weak_7x7 - weak_7x7.mean(dim=1, keepdim=True)
    native7x7 = F.cosine_similarity(u7, v7, dim=1, eps=1e-8)
    native7x7_n = (native7x7 - native7x7.min()) / (native7x7.max() - native7x7.min() + 1e-8)
    
    # 5: Spatial Entropy Certainty Factor
    p_attn = raw_attn_7x7.view(B, -1)
    p_attn = p_attn / (p_attn.sum(dim=-1, keepdim=True) + 1e-8)
    spat_entropy = -(p_attn * (p_attn + 1e-8).log()).sum(dim=-1)
    max_h = math.log(49)
    spat_certainty = torch.clamp((max_h - spat_entropy) / (max_h * 0.35), 0.0, 1.0)
    
    score_dict = {}
    for cfg in configs:
        cname = cfg["name"]
        ctype = cfg["type"]
        if ctype == "baseline":
            score_dict[cname] = ss_base
        elif ctype == "pearson":
            w = cfg["w"]
            if cfg["gate"] == "linear":
                score_dict[cname] = ss_base * ((1.0 - w) + w * spat_pearson_n)
            elif cfg["gate"] == "exp":
                score_dict[cname] = ss_base * torch.exp(w * (spat_pearson_n - 1.0))
        elif ctype == "js":
            w = cfg["w"]
            score_dict[cname] = ss_base * ((1.0 - w) + w * spat_js_n)
        elif ctype == "sal_pearson":
            w = cfg["w"]
            score_dict[cname] = ss_base * ((1.0 - w) + w * sal_pearson_n)
        elif ctype == "native7x7":
            w = cfg["w"]
            score_dict[cname] = ss_base * ((1.0 - w) + w * native7x7_n)
        elif ctype == "entropy_mod":
            w = cfg["w"]
            eff_w = w * spat_certainty
            score_dict[cname] = ss_base * ((1.0 - eff_w) + eff_w * spat_pearson_n)

    tau = 0.25
    for cname, score_vec in score_dict.items():
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        out = model(images)
        ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)
        sel = torch.topk(ent, 3, largest=False).indices
        w_c = F.softmax(score_vec[sel] / tau, dim=-1)
        loss = entropy_avg(out[sel]) + 1.0 * self_consistency_loss(out[sel], w_c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = (F.softmax(score_vec / tau, dim=-1).unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            if pred == target:
                counts[cname] += 1
                
    total += 1
    if (i + 1) % 50 == 0 or (i + 1) == args.samples:
        elapsed = time.time() - t0
        base_acc = counts["baseline"] / total * 100.0
        best_cfg = max(counts, key=counts.get)
        best_acc = counts[best_cfg] / total * 100.0
        print(f"[{args.domain.upper()}] Sample {i+1}/{args.samples} ({elapsed:.1f}s) | Base: {base_acc:.2f}% | Best: {best_cfg} ({best_acc:.2f}%)", flush=True)

final_results = {}
base_acc = counts["baseline"] / total * 100.0
for k, v in counts.items():
    acc = v / total * 100.0
    final_results[k] = {"acc": acc, "diff": acc - base_acc, "correct": v, "total": total}

out_file = f"task1_exp_{args.domain}.json"
with open(out_file, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\n[{args.domain.upper()}] Exploration Finished! Results saved to {out_file}.")

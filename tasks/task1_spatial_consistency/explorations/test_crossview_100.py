import subprocess
import os
import re
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np
import time
import torchvision.transforms as transforms
from clip import clip
from clip.custom_clip import get_coop
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import caltech101_classes
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
import torchattacks
from sstpt import (
    TinyAugmenterWithTransform,
    compute_stability_and_spatial_consistency,
    select_confident_views,
    entropy_avg,
    self_consistency_loss_with_rs_view,
    AverageMeter,
    accuracy
)

device = torch.device("cuda:1")
arch = "RN50"
DATA_PATH = "/home/abhishek/workspace/data"
num_samples = 100

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
preprocess = transforms.Compose([transforms.ToTensor()])
data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=True)

# Build ImageNet-R classes
label_mask = imagenet_r_mask
r_classes = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]

experiments = [
    {"name": "Clean (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 0.0, "steps": 0, "alpha": 0.0},
    {"name": "Adversarial PGD (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 1.0, "steps": 7, "alpha": 0.25},
    {"name": "Distribution-Shift (ImageNet-R)", "dataset": "R", "classes": r_classes, "eps": 0.0, "steps": 0, "alpha": 0.0},
]

def run_eval(exp, use_cross_spat=False, spat_weight=0.20):
    dset_name = exp["dataset"]
    classnames = exp["classes"]
    eps = exp["eps"]
    steps = exp["steps"]
    alpha = exp["alpha"]
    
    val_dataset = build_dataset(dset_name, data_transform, DATA_PATH, mode="test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    model = get_coop(arch, classnames, str(device), 4, "a_photo_of_a", learned_cls=False)
    model.eval()
    
    # Store initial model state
    initial_ctx = model.prompt_learner.ctx.clone().detach()
    
    atk = torchattacks.PGD(model, eps=eps/255.0, alpha=alpha/255.0, steps=steps) if eps > 0 else None
    tiny_aug = TinyAugmenterWithTransform()
    
    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = F.normalize(text_features, dim=1)
        logit_scale = model.logit_scale.exp()
        
    top1 = AverageMeter('Acc@1', ':6.2f')
    t0 = time.time()
    
    for i, (images, target) in enumerate(val_loader):
        if i >= num_samples:
            break
        target = target.to(device, non_blocking=True)
        
        if eps > 0.0:
            image = images[0].to(device, non_blocking=True)
            adv_image = atk(image, target)
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=True)
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=True)
        images = torch.cat(images, dim=0) # [16, 3, 224, 224]
        
        # Reset model prompt
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
            
        # Optimizer for TTA
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, lr=5e-3)
        
        # Extract features and spatial attention maps
        if use_cross_spat:
            model.image_encoder.attnpool.return_attn = True
            with torch.no_grad():
                clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_outputs = logit_scale * clip_features @ text_features.t()
            model.image_encoder.attnpool.return_attn = False
            
            # Cross-View Spatial Maps [16, 49]
            attn = model.image_encoder.attnpool.last_attn[:, 0, 1:]
            spatial_centered = attn - attn.mean(dim=-1, keepdim=True)
            spatial_norm = F.normalize(spatial_centered, p=2, dim=-1)
            S_spat = torch.mm(spatial_norm, spatial_norm.t()) # [16, 16]
            consensus = (S_spat.sum(dim=1) - 1.0) / 15.0
            base_sim = S_spat[:, 0]
            spat_score = 0.5 * consensus + 0.5 * base_sim
        else:
            with torch.no_grad():
                clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                clip_outputs = logit_scale * clip_features @ text_features.t()
            spat_score = None
            
        # Suitability score
        suit_k = 15
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        dist = 2 - 2 * sim_matrix
        knn_dists, _ = dist.topk(suit_k+1, dim=-1, largest=False)
        suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8)
        suit_score = suit_score.squeeze(0)
        
        # Stability score
        stab_score, _ = compute_stability_and_spatial_consistency(
            model, images, base_logits=clip_outputs, aug_builder=tiny_aug, K=1, use_spatial=False
        )
        
        # Combined score calculation
        suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
        stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
        
        ss_base = 0.40 * suit_n + 0.60 * stab_n
        if use_cross_spat and spat_score is not None:
            spat_n = (spat_score - spat_score.min()) / (spat_score.max() - spat_score.min() + 1e-8)
            # Multiplicative Cross-View Gating
            score = ss_base * ((1.0 - spat_weight) + spat_weight * spat_n)
        else:
            score = ss_base
            
        # Test-time tuning (1 step)
        output_all = model(images)
        output, selected_idx = select_confident_views(output_all, 3)
        weight_cons = F.softmax(score[selected_idx] / 0.25, dim=-1)
        
        ent_loss = entropy_avg(output)
        kl_loss = self_consistency_loss_with_rs_view(output, weight_cons, "weighted")
        loss = ent_loss + 1.0 * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Final prediction ensemble
        with torch.no_grad():
            tuned_outputs = model(images)
            
        weight_pred = F.softmax(score / 0.25, dim=-1)
        tta_output = torch.bmm(weight_pred.unsqueeze(0).unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)
        
        acc1, _ = accuracy(tta_output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))
        
    eval_time = time.time() - t0
    return top1.avg, eval_time

print(f"==========================================================================")
print(f"Running 100-Sample Evaluation: Baseline SS-TPT vs. Cross-View Spatial S3-TPT")
print(f"==========================================================================")

results = []
for exp in experiments:
    print(f"\n--- Testing on {exp['name']} ({num_samples} samples) ---")
    
    # 1. Baseline SS-TPT
    print("Evaluating Baseline SS-TPT (Suitability + Stability)...")
    acc_base, time_base = run_eval(exp, use_cross_spat=False)
    print(f" -> Baseline Top-1 Acc: {acc_base:.2f}% (Time: {time_base:.1f}s)")
    
    # 2. Cross-View Spatial S3-TPT (w_spat = 0.20)
    print("Evaluating Proposed Cross-View S3-TPT (w_spat=0.20)...")
    acc_cross, time_cross = run_eval(exp, use_cross_spat=True, spat_weight=0.20)
    print(f" -> Cross-View S3-TPT Top-1 Acc: {acc_cross:.2f}% (Time: {time_cross:.1f}s)")
    
    diff = acc_cross - acc_base
    results.append({
        "domain": exp["name"],
        "baseline_acc": acc_base,
        "cross_acc": acc_cross,
        "diff": diff,
        "time_base": time_base,
        "time_cross": time_cross
    })

print(f"\n\n==========================================================================")
print(f"{'Domain':<30} | {'Baseline SS-TPT':<16} | {'Cross-View S3-TPT':<18} | {'Diff':<8}")
print(f"--------------------------------------------------------------------------")
for r in results:
    sign = "+" if r['diff'] >= 0 else ""
    print(f"{r['domain']:<30} | {r['baseline_acc']:>6.2f}%          | {r['cross_acc']:>6.2f}%           | {sign}{r['diff']:>5.2f}%")
print(f"==========================================================================")

import torch
import torch.nn.functional as F
import numpy as np
import time
from data.datautils import build_dataset, AugMixAugmenter
from torchvision import transforms
from clip.custom_clip import get_coop
from data.cls_to_names import caltech101_classes
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
from sstpt import TinyAugmenterWithTransform, js_divergence
import torchattacks

device = "cuda:1"
num_samples = 100

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
preprocess = transforms.Compose([transforms.ToTensor()])
data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=True)

label_mask = imagenet_r_mask
r_classes = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]

experiments = [
    {"name": "Clean (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 0.0, "steps": 0, "alpha": 0.0},
    {"name": "Adversarial PGD (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 1.0, "steps": 7, "alpha": 0.25},
    {"name": "Distribution-Shift (ImageNet-R)", "dataset": "R", "classes": r_classes, "eps": 0.0, "steps": 0, "alpha": 0.0},
]

def entropy_avg(outputs):
    return (-(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()

def self_consistency_loss(outputs, weight):
    probs = outputs.softmax(1).clamp_min(1e-8)
    p_bar = (weight.unsqueeze(1) * probs).sum(0, keepdim=True)
    return F.kl_div(outputs.log_softmax(1), p_bar.repeat(outputs.size(0), 1), reduction='batchmean')

tiny_aug = TinyAugmenterWithTransform()

def eval_metric(exp, spat_type="pearson", w_spat=0.15, selection="entropy"):
    dset_name = exp["dataset"]
    classnames = exp["classes"]
    eps = exp["eps"]
    steps = exp["steps"]
    alpha = exp["alpha"]
    
    val_dataset = build_dataset(dset_name, data_transform, "/home/abhishek/workspace/data", mode="test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    model = get_coop("RN50", classnames, str(device), 4, "a_photo_of_a", learned_cls=False)
    model.eval()
    initial_ctx = model.prompt_learner.ctx.clone().detach()
    atk = torchattacks.PGD(model, eps=eps/255.0, alpha=alpha/255.0, steps=steps) if eps > 0 else None
    
    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(), dim=1)
        logit_scale = model.logit_scale.exp()
        
    correct = 0
    total = 0
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

        images = torch.cat([img.to(device, non_blocking=True) for img in images], dim=0) # [16, 3, 224, 224]
        
        # 1. Base forward pass & spatial maps
        base_map = model.get_spatial_relevance_map(images, method='attn') # [16, 1, 224, 224]
        
        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()
            
        # 2. Weak augmentation (K=1) for stability and spatial consistency
        aug_views, t_params = tiny_aug(images)
        with torch.no_grad():
            q_logits = model(aug_views)
        d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1)) # [16]
        stab_score = 1.0 / (d + 1e-8)
        
        # Weak spatial map
        weak_map = model.get_spatial_relevance_map(aug_views, method='attn') # [16, 1, 224, 224]
        aligned_base_map = tiny_aug.align_spatial_map(base_map, t_params)
        
        # Crop borders to remove boundary padding artifacts
        margin = 16
        m_base = aligned_base_map[:, :, margin:-margin, margin:-margin]
        m_weak = weak_map[:, :, margin:-margin, margin:-margin]
        
        # Compute spatial consistency score:
        if spat_type == "raw_cosine":
            spat_score = F.cosine_similarity(m_base.flatten(1), m_weak.flatten(1), dim=1, eps=1e-8)
        elif spat_type == "pearson":
            # Pearson correlation (zero-mean centering)
            u = m_base.flatten(1)
            v = m_weak.flatten(1)
            u_c = u - u.mean(dim=1, keepdim=True)
            v_c = v - v.mean(dim=1, keepdim=True)
            spat_score = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8)
        elif spat_type == "iou":
            # Saliency IoU above mean
            u = m_base.flatten(1)
            v = m_weak.flatten(1)
            u_bin = (u > u.mean(dim=1, keepdim=True)).float()
            v_bin = (v > v.mean(dim=1, keepdim=True)).float()
            intersection = (u_bin * v_bin).sum(dim=1)
            union = ((u_bin + v_bin) > 0).float().sum(dim=1)
            spat_score = intersection / (union + 1e-8)
        elif spat_type == "none":
            spat_score = None
            
        # Suitability score
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        dist = 2 - 2 * sim_matrix
        knn_dists, _ = dist.topk(16, dim=-1, largest=False)
        suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
        
        # Normalize scores
        suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
        stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
        
        if spat_score is not None:
            spat_n = (spat_score - spat_score.min()) / (spat_score.max() - spat_score.min() + 1e-8)
            # Combine scores: preserving suit:stab 0.4:0.6 ratio
            # score = (1 - w_spat) * (0.4 * suit_n + 0.6 * stab_n) + w_spat * spat_n
            score = (1.0 - w_spat) * (0.40 * suit_n + 0.60 * stab_n) + w_spat * spat_n
        else:
            score = 0.40 * suit_n + 0.60 * stab_n
            
        # Test-time tuning
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        
        output_all = model(images)
        
        if selection == "entropy":
            entropy = -(output_all.softmax(1) * output_all.log_softmax(1)).sum(1)
            selected_idx = torch.topk(entropy, 3, largest=False).indices
        elif selection == "quality_pareto":
            # Top 8 by quality score, then top 3 by entropy
            cand = torch.topk(score, 8, largest=True).indices
            cand_ent = -(output_all[cand].softmax(1) * output_all[cand].log_softmax(1)).sum(1)
            best_cand = torch.topk(cand_ent, 3, largest=False).indices
            selected_idx = cand[best_cand]
            
        weight_cons = F.softmax(score[selected_idx] / 0.25, dim=-1)
        loss = entropy_avg(output_all[selected_idx]) + 1.0 * self_consistency_loss(output_all[selected_idx], weight_cons)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            tuned_outputs = model(images)
            
        weight_pred = F.softmax(score / 0.25, dim=-1)
        tta_output = (weight_pred.unsqueeze(-1) * tuned_outputs).sum(0, keepdim=True)
        pred = tta_output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1
        
    acc = correct / total * 100.0
    return acc, time.time() - t0

print("=" * 80)
print("Evaluating Task 1 Formulations on 100 Samples across All 3 Domains")
print("=" * 80)

for exp in experiments:
    print(f"\n>>> {exp['name']} <<<", flush=True)
    
    # 1. Baseline SS-TPT
    acc_base, t_b = eval_metric(exp, spat_type="none", w_spat=0.0, selection="entropy")
    print(f"  [1] Original 2-Score Baseline: {acc_base:.2f}% ({t_b:.1f}s)", flush=True)
    
    # 2. Raw Cosine (earlier implementation)
    acc_cos, t_cos = eval_metric(exp, spat_type="raw_cosine", w_spat=0.15, selection="entropy")
    diff_cos = acc_cos - acc_base
    print(f"  [2] Raw Cosine (w=0.15):       {acc_cos:.2f}% ({'+' if diff_cos>=0 else ''}{diff_cos:.2f}%) ({t_cos:.1f}s)", flush=True)
    
    # 3. Pearson Correlation (Zero-Mean Centered)
    acc_pearson, t_p = eval_metric(exp, spat_type="pearson", w_spat=0.15, selection="entropy")
    diff_p = acc_pearson - acc_base
    print(f"  [3] Pearson Correlation:       {acc_pearson:.2f}% ({'+' if diff_p>=0 else ''}{diff_p:.2f}%) ({t_p:.1f}s)", flush=True)
    
    # 4. Saliency IoU
    acc_iou, t_iou = eval_metric(exp, spat_type="iou", w_spat=0.15, selection="entropy")
    diff_iou = acc_iou - acc_base
    print(f"  [4] Saliency IoU:              {acc_iou:.2f}% ({'+' if diff_iou>=0 else ''}{diff_iou:.2f}%) ({t_iou:.1f}s)", flush=True)
    
    # 5. Pearson + Quality Pareto Selection
    acc_pareto, t_par = eval_metric(exp, spat_type="pearson", w_spat=0.15, selection="quality_pareto")
    diff_par = acc_pareto - acc_base
    print(f"  [5] Pearson + Quality Pareto:  {acc_pareto:.2f}% ({'+' if diff_par>=0 else ''}{diff_par:.2f}%) ({t_par:.1f}s)", flush=True)

print("\nBenchmark completed successfully!")

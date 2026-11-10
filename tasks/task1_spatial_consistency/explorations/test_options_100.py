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

def run_test(exp, mode="baseline", keep_top_k=8):
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

        if isinstance(images, list):
            images = [img.to(device, non_blocking=True) for img in images]
        else:
            if len(images.size()) > 4:
                images = images.squeeze(0)
            images = images.to(device, non_blocking=True)
        images = torch.cat(images, dim=0) # [16, 3, 224, 224]
        
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        
        # 1. Forward features & spatial maps
        model.image_encoder.attnpool.return_attn = True
        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()
        model.image_encoder.attnpool.return_attn = False
        
        # Cross-view spatial score
        attn = model.image_encoder.attnpool.last_attn[:, 0, 1:] # [16, 49]
        spatial_centered = attn - attn.mean(dim=-1, keepdim=True)
        spatial_norm = F.normalize(spatial_centered, p=2, dim=-1)
        S_spat = torch.mm(spatial_norm, spatial_norm.t()) # [16, 16]
        consensus = (S_spat.sum(dim=1) - 1.0) / 15.0
        spat_score = 0.5 * consensus + 0.5 * S_spat[:, 0]
        
        # Suitability score
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        dist = 2 - 2 * sim_matrix
        knn_dists, _ = dist.topk(16, dim=-1, largest=False)
        suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
        suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
        
        output_all = model(images)
        
        if mode == "baseline":
            # Pure entropy selection
            entropy = -(output_all.softmax(1) * output_all.log_softmax(1)).sum(1)
            selected_idx = torch.topk(entropy, 3, largest=False).indices
            weight_cons = F.softmax(suit_n[selected_idx] / 0.25, dim=-1)
            
            ent_loss = entropy_avg(output_all[selected_idx])
            kl_loss = self_consistency_loss(output_all[selected_idx], weight_cons)
            loss = ent_loss + 1.0 * kl_loss
            
        elif mode == "option_a":
            # Option A: Spatial Filtering before entropy selection
            spat_candidates = torch.topk(spat_score, keep_top_k, largest=True).indices
            cand_outputs = output_all[spat_candidates]
            cand_entropy = -(cand_outputs.softmax(1) * cand_outputs.log_softmax(1)).sum(1)
            best_cand = torch.topk(cand_entropy, 3, largest=False).indices
            selected_idx = spat_candidates[best_cand]
            
            weight_cons = F.softmax(suit_n[selected_idx] / 0.25, dim=-1)
            ent_loss = entropy_avg(output_all[selected_idx])
            kl_loss = self_consistency_loss(output_all[selected_idx], weight_cons)
            loss = ent_loss + 1.0 * kl_loss
            
        elif mode == "option_b":
            # Option B: Spatial Alignment Loss
            entropy = -(output_all.softmax(1) * output_all.log_softmax(1)).sum(1)
            selected_idx = torch.topk(entropy, 3, largest=False).indices
            weight_cons = F.softmax(suit_n[selected_idx] / 0.25, dim=-1)
            
            ent_loss = entropy_avg(output_all[selected_idx])
            kl_loss = self_consistency_loss(output_all[selected_idx], weight_cons)
            
            curr_text_features = F.normalize(model.get_text_features(), dim=1)
            pred_classes = output_all[selected_idx].argmax(dim=1)
            selected_text = curr_text_features[pred_classes]
            spat_weights = F.softmax(spat_score[selected_idx] / 0.25, dim=-1)
            align_loss = -torch.sum(spat_weights * (selected_text * clip_features[selected_idx]).sum(dim=-1))
            
            loss = ent_loss + 1.0 * kl_loss + 0.1 * align_loss

        elif mode == "option_ab":
            # Combination Option A + B
            spat_candidates = torch.topk(spat_score, keep_top_k, largest=True).indices
            cand_outputs = output_all[spat_candidates]
            cand_entropy = -(cand_outputs.softmax(1) * cand_outputs.log_softmax(1)).sum(1)
            best_cand = torch.topk(cand_entropy, 3, largest=False).indices
            selected_idx = spat_candidates[best_cand]
            
            weight_cons = F.softmax(suit_n[selected_idx] / 0.25, dim=-1)
            ent_loss = entropy_avg(output_all[selected_idx])
            kl_loss = self_consistency_loss(output_all[selected_idx], weight_cons)
            
            curr_text_features = F.normalize(model.get_text_features(), dim=1)
            pred_classes = output_all[selected_idx].argmax(dim=1)
            selected_text = curr_text_features[pred_classes]
            spat_weights = F.softmax(spat_score[selected_idx] / 0.25, dim=-1)
            align_loss = -torch.sum(spat_weights * (selected_text * clip_features[selected_idx]).sum(dim=-1))
            
            loss = ent_loss + 1.0 * kl_loss + 0.1 * align_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            tuned_outputs = model(images)
            
        if mode in ["option_a", "option_ab"]:
            valid_mask = torch.zeros(16, device=device)
            valid_mask[spat_candidates] = 1.0
            score_ensemble = suit_n * valid_mask
            weight_pred = F.softmax(score_ensemble / 0.25, dim=-1)
        else:
            weight_pred = F.softmax(suit_n / 0.25, dim=-1)
            
        tta_output = (weight_pred.unsqueeze(-1) * tuned_outputs).sum(0, keepdim=True)
        pred = tta_output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1
        
    acc = correct / total * 100.0
    return acc, time.time() - t0

print(f"==========================================================================")
print(f"100-Sample Evaluation: Baseline vs Option A vs Option B vs Option A+B")
print(f"==========================================================================")

for exp in experiments:
    print(f"\n>>> Dataset: {exp['name']} ({num_samples} samples) <<<")
    
    acc_base, t_base = run_test(exp, mode="baseline")
    print(f"Baseline SS-TPT:   Top-1 Acc = {acc_base:.2f}% ({t_base:.1f}s)")
    
    acc_a, t_a = run_test(exp, mode="option_a", keep_top_k=8)
    diff_a = acc_a - acc_base
    print(f"Option A (Filter): Top-1 Acc = {acc_a:.2f}% ({'+' if diff_a>=0 else ''}{diff_a:.2f}%) ({t_a:.1f}s)")
    
    acc_b, t_b = run_test(exp, mode="option_b")
    diff_b = acc_b - acc_base
    print(f"Option B (Align):  Top-1 Acc = {acc_b:.2f}% ({'+' if diff_b>=0 else ''}{diff_b:.2f}%) ({t_b:.1f}s)")

    acc_ab, t_ab = run_test(exp, mode="option_ab", keep_top_k=8)
    diff_ab = acc_ab - acc_base
    print(f"Option A+B Combo:  Top-1 Acc = {acc_ab:.2f}% ({'+' if diff_ab>=0 else ''}{diff_ab:.2f}%) ({t_ab:.1f}s)")

print("\nEvaluation completed!")

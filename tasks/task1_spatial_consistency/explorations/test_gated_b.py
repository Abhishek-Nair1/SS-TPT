import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
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

def run_test(exp):
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
        
    correct_base = 0
    correct_gated = 0
    total = 0
    
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
        
        model.image_encoder.attnpool.return_attn = True
        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()
        model.image_encoder.attnpool.return_attn = False
        
        attn = model.image_encoder.attnpool.last_attn[:, 0, 1:]
        spatial_centered = attn - attn.mean(dim=-1, keepdim=True)
        spatial_norm = F.normalize(spatial_centered, p=2, dim=-1)
        S_spat = torch.mm(spatial_norm, spatial_norm.t())
        consensus = (S_spat.sum(dim=1) - 1.0) / 15.0
        spat_score = 0.5 * consensus + 0.5 * S_spat[:, 0]
        
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        dist = 2 - 2 * sim_matrix
        knn_dists, _ = dist.topk(16, dim=-1, largest=False)
        suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
        suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
        
        # Stability divergence
        aug_views, _ = tiny_aug(images)
        with torch.no_grad():
            q_logits = model(aug_views)
        d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
        mean_div = d.mean().item()
        
        # Baseline
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt_b = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        out_b = model(images)
        ent_b = -(out_b.softmax(1) * out_b.log_softmax(1)).sum(1)
        sel_idx = torch.topk(ent_b, 3, largest=False).indices
        w_cons = F.softmax(suit_n[sel_idx] / 0.25, dim=-1)
        loss_b = entropy_avg(out_b[sel_idx]) + 1.0 * self_consistency_loss(out_b[sel_idx], w_cons)
        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()
        with torch.no_grad():
            pred_b = (F.softmax(suit_n / 0.25, dim=-1).unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            correct_base += (pred_b == target).sum().item()
            
        # Gated Option B
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt_g = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        out_g = model(images)
        ent_g = -(out_g.softmax(1) * out_g.log_softmax(1)).sum(1)
        sel_idx_g = torch.topk(ent_g, 3, largest=False).indices
        w_cons_g = F.softmax(suit_n[sel_idx_g] / 0.25, dim=-1)
        
        curr_text = F.normalize(model.get_text_features(), dim=1)
        pred_cls = out_g[sel_idx_g].argmax(dim=1)
        sel_text = curr_text[pred_cls]
        spat_w = F.softmax(spat_score[sel_idx_g] / 0.25, dim=-1)
        align_loss = -torch.sum(spat_w * (sel_text * clip_features[sel_idx_g]).sum(dim=-1))
        
        # Stability gate: only align when predictions are stable (clean / domain-shifted)
        # In adv, mean_div > 0.05
        gate = 1.0 if mean_div < 0.04 else 0.0
        loss_g = entropy_avg(out_g[sel_idx_g]) + 1.0 * self_consistency_loss(out_g[sel_idx_g], w_cons_g) + (0.05 * gate) * align_loss
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        with torch.no_grad():
            pred_g = (F.softmax(suit_n / 0.25, dim=-1).unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            correct_gated += (pred_g == target).sum().item()
            
        total += 1
        if (i + 1) % 25 == 0:
            print(f"[{exp['name']}] Sample {i+1}/100: Base={correct_base/(i+1)*100:.1f}%, Gated={correct_gated/(i+1)*100:.1f}%", flush=True)
            
    acc_b = correct_base / total * 100.0
    acc_g = correct_gated / total * 100.0
    return acc_b, acc_g

for exp in experiments:
    print(f"\nEvaluating {exp['name']}...", flush=True)
    t0 = time.time()
    acc_b, acc_g = run_test(exp)
    diff = acc_g - acc_b
    print(f"--> {exp['name']}: Base = {acc_b:.2f}% | Stability-Gated Option B = {acc_g:.2f}% ({'+' if diff>=0 else ''}{diff:.2f}%) [{time.time()-t0:.1f}s]", flush=True)


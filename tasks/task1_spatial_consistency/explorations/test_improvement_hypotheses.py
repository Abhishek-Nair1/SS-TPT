import torch
import torch.nn.functional as F
import numpy as np
import time
from data.datautils import build_dataset, AugMixAugmenter
from torchvision import transforms
import torchvision.transforms.functional as TF
from clip.custom_clip import get_coop
from data.cls_to_names import caltech101_classes
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
from sstpt import TinyAugmenterWithTransform, js_divergence
import torchattacks

device = "cuda:1"
num_samples = 150

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
preprocess = transforms.Compose([transforms.ToTensor()])

label_mask = imagenet_r_mask
r_classes = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]

experiments = [
    {"name": "Distribution-Shift (ImageNet-R)", "dataset": "R", "classes": r_classes, "eps": 0.0, "steps": 0, "alpha": 0.0, "augmix": False},
    {"name": "Clean (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 0.0, "steps": 0, "alpha": 0.0, "augmix": True},
    {"name": "Adversarial PGD (Caltech101)", "dataset": "Caltech101", "classes": caltech101_classes, "eps": 1.0, "steps": 7, "alpha": 0.25, "augmix": True},
]

def entropy_avg(outputs):
    return (-(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)).mean()

def self_consistency_loss(outputs, weight):
    probs = outputs.softmax(1).clamp_min(1e-8)
    p_bar = (weight.unsqueeze(1) * probs).sum(0, keepdim=True)
    return F.kl_div(outputs.log_softmax(1), p_bar.repeat(outputs.size(0), 1), reduction='batchmean')

tiny_aug = TinyAugmenterWithTransform()

def run_sample(model, images, target, initial_ctx, text_features, logit_scale):
    # Base forward
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()
    model.image_encoder.attnpool.return_attn = False
    
    # 7x7 attention map: [B, 49] -> [B, 1, 7, 7]
    attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    
    # Suitability
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(16, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8).squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)

    # Weak augmentation
    aug_views, t_params = tiny_aug(images)
    with torch.no_grad():
        q_logits = model(aug_views)
    d = js_divergence(clip_outputs.softmax(-1), q_logits.softmax(-1))
    stab_score = 1.0 / (d + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
    ss_base = 0.40 * suit_n + 0.60 * stab_n

    # 1. Cosine 224x224 (Old method)
    base_map_224 = F.interpolate(attn_7x7, size=(224, 224), mode='bilinear', align_corners=False)
    # Weak map
    model.image_encoder.attnpool.return_attn = True
    with torch.no_grad():
        _ = model.image_encoder(model.normalize(aug_views.type(model.dtype)))
    model.image_encoder.attnpool.return_attn = False
    weak_attn_7x7 = model.image_encoder.attnpool.last_attn[:, 0, 1:].view(-1, 1, 7, 7)
    weak_map_224 = F.interpolate(weak_attn_7x7, size=(224, 224), mode='bilinear', align_corners=False)
    
    aligned_base_224 = tiny_aug.align_spatial_map(base_map_224, t_params)
    m_base = aligned_base_224[:, :, 16:-16, 16:-16]
    m_weak = weak_map_224[:, :, 16:-16, 16:-16]
    spat_cos = F.cosine_similarity(m_base.flatten(1), m_weak.flatten(1), dim=1, eps=1e-8)
    spat_cos_n = (spat_cos - spat_cos.min()) / (spat_cos.max() - spat_cos.min() + 1e-8)
    
    # 2. Pearson 224x224
    u = m_base.flatten(1)
    v = m_weak.flatten(1)
    u_c = u - u.mean(dim=1, keepdim=True)
    v_c = v - v.mean(dim=1, keepdim=True)
    spat_pearson = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8)
    spat_pearson_n = (spat_pearson - spat_pearson.min()) / (spat_pearson.max() - spat_pearson.min() + 1e-8)
    
    # 3. Native 7x7 Pearson (without interpolation blur)
    aligned_base_7x7 = TF.affine(
        attn_7x7,
        angle=t_params['angle'],
        translate=[t_params['translate'][0] * 7.0 / 224.0, t_params['translate'][1] * 7.0 / 224.0],
        scale=t_params['scale'],
        shear=[0.0, 0.0],
        interpolation=transforms.InterpolationMode.BILINEAR
    )
    u7 = aligned_base_7x7.flatten(1)
    v7 = weak_attn_7x7.flatten(1)
    u7_c = u7 - u7.mean(dim=1, keepdim=True)
    v7_c = v7 - v7.mean(dim=1, keepdim=True)
    spat_native = F.cosine_similarity(u7_c, v7_c, dim=1, eps=1e-8)
    spat_native_n = (spat_native - spat_native.min()) / (spat_native.max() - spat_native.min() + 1e-8)

    # 4. Cross-View Spatial Consensus (Task 1 Section 15.1: corrupted views are inconsistent across views)
    spatial_centered = attn_7x7.flatten(1) - attn_7x7.flatten(1).mean(dim=-1, keepdim=True)
    spatial_norm = F.normalize(spatial_centered, p=2, dim=-1)
    S_mat = torch.mm(spatial_norm, spatial_norm.t()) # [16, 16]
    consensus = (S_mat.sum(dim=1) - 1.0) / 15.0
    cons_n = (consensus - consensus.min()) / (consensus.max() - consensus.min() + 1e-8)
    
    # Combined Multi-Aspect Spatial Score: temporal weak alignment + cross-view consensus
    spat_multi = 0.5 * spat_pearson_n + 0.5 * cons_n
    spat_multi_n = (spat_multi - spat_multi.min()) / (spat_multi.max() - spat_multi.min() + 1e-8)

    scores = {
        "base": ss_base,
        "gated_cos": ss_base * (0.90 + 0.10 * spat_cos_n),
        "gated_pearson": ss_base * (0.90 + 0.10 * spat_pearson_n),
        "gated_native7x7": ss_base * (0.90 + 0.10 * spat_native_n),
        "gated_multi": ss_base * (0.90 + 0.10 * spat_multi_n),
    }

    # Evaluate each method
    corrects = {}
    for name, s in scores.items():
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(initial_ctx)
        opt = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
        out = model(images)
        ent = -(out.softmax(1) * out.log_softmax(1)).sum(1)
        sel = torch.topk(ent, 3, largest=False).indices
        w_c = F.softmax(s[sel] / 0.25, dim=-1)
        loss = entropy_avg(out[sel]) + 1.0 * self_consistency_loss(out[sel], w_c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = (F.softmax(s / 0.25, dim=-1).unsqueeze(-1) * model(images)).sum(0, keepdim=True).argmax(-1)
            corrects[name] = int(pred == target)
            
    return corrects

print("Starting Hypotheses Evaluation on 150 Samples per Domain...")

def eval_experiment(exp):
    dset_name = exp["dataset"]
    classnames = exp["classes"]
    eps = exp["eps"]
    steps = exp["steps"]
    alpha = exp["alpha"]
    augmix = exp["augmix"]
    
    dt = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=augmix)
    val_dataset = build_dataset(dset_name, dt, "/home/abhishek/workspace/data", mode="test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_coop("RN50", classnames, str(device), 4, "a_photo_of_a", learned_cls=False)
    model.eval()
    initial_ctx = model.prompt_learner.ctx.clone().detach()
    atk = torchattacks.PGD(model, eps=eps/255.0, alpha=alpha/255.0, steps=steps) if eps > 0 else None
    
    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(), dim=1)
        logit_scale = model.logit_scale.exp()
        
    counts = {"base": 0, "gated_cos": 0, "gated_pearson": 0, "gated_native7x7": 0, "gated_multi": 0}
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
            images = dt(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        images = torch.cat([img.to(device, non_blocking=True) for img in images], dim=0)
        
        c = run_sample(model, images, target, initial_ctx, text_features, logit_scale)
        for k in counts:
            counts[k] += c[k]
        total += 1
        
        if (i + 1) % 50 == 0:
            print(f"[{exp['name']}] Sample {i+1}/{num_samples}: Base={counts['base']/(i+1)*100:.1f}%, Pearson={counts['gated_pearson']/(i+1)*100:.1f}%, Native7x7={counts['gated_native7x7']/(i+1)*100:.1f}%, Multi={counts['gated_multi']/(i+1)*100:.1f}%", flush=True)
            
    print(f"\n>>> Results for {exp['name']} ({total} samples, {time.time()-t0:.1f}s) <<<", flush=True)
    base_acc = counts['base'] / total * 100.0
    for k in counts:
        acc = counts[k] / total * 100.0
        diff = acc - base_acc
        print(f"  {k:<16}: {acc:.2f}% ({'+' if diff>=0 else ''}{diff:.2f}%)", flush=True)

for exp in experiments:
    eval_experiment(exp)


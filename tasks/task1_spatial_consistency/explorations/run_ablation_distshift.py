import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from clip.custom_clip import get_coop
from data.datautils import AugMixAugmenter, build_dataset
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_r_mask
from sstpt import TinyAugmenterWithTransform, js_divergence, self_consistency_loss_with_rs_view, entropy_avg

device = 'cuda:1'
classnames_all = imagenet_classes
classnames = [classnames_all[i] for i, m in enumerate(imagenet_r_mask) if m]
model = get_coop('RN50', classnames, str(device), n_ctx=4, ctx_init='a_photo_of_a').to(device)

base_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224)
])
preprocess = transforms.Compose([transforms.ToTensor()])
data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=True)

val_dataset = build_dataset('R', data_transform, '/home/abhishek/workspace/data', mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

tiny_aug = TinyAugmenterWithTransform()

N_SAMPLES = 50
print(f"Comparing strategies on {N_SAMPLES} ImageNet-R (Distribution-Shift) samples...")

strategies = [
    "Baseline SS-TPT (Entropy Selection)",
    "S3-TPT (Score-Guided View Selection, Grad-CAM)"
]

correct_counts = {s: 0 for s in strategies}
total = 0

trainable_param = model.prompt_learner.parameters()
optim_state_base = torch.optim.AdamW(trainable_param, 5e-3).state_dict()

with torch.no_grad():
    text_features = model.get_text_features()
    text_features = F.normalize(text_features, dim=1)
    logit_scale = model.logit_scale.exp()

for i, (images, target) in enumerate(val_loader):
    if i >= N_SAMPLES:
        break
    target = target.to(device, non_blocking=True)
    images = torch.cat(images, dim=0).to(device)

    # 1. Feature extraction & Suitability
    with torch.no_grad():
        clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        clip_outputs = logit_scale * clip_features @ text_features.t()

    suit_k = images.shape[0] - 1
    sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
    dist = 2 - 2 * sim_matrix
    knn_dists, _ = dist.topk(suit_k+1, dim=-1, largest=False)
    suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8)
    suit_score = suit_score.squeeze(0)
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)

    # 2. Stability & Spatial Consistency
    p = clip_outputs.softmax(dim=-1)
    cam_base = model.get_spatial_relevance_map(images, method='gradcam')

    divergences = []
    spatial_sims = []
    for _ in range(1):
        aug_views, t_params = tiny_aug(images)
        with torch.no_grad():
            q_logits = model(aug_views)
        q = q_logits.softmax(dim=-1)
        divergences.append(js_divergence(p, q))
        
        cam_weak = model.get_spatial_relevance_map(aug_views, method='gradcam')
        cam_aligned = tiny_aug.align_spatial_map(cam_base, t_params)
        sim = F.cosine_similarity(cam_aligned.flatten(1), cam_weak.flatten(1), dim=1, eps=1e-8)
        spatial_sims.append(sim)

    div = torch.stack(divergences, dim=0).mean(0)
    stab_score = 1.0 / (div + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)

    spat_score = torch.stack(spatial_sims, dim=0).mean(0)
    spat_n = (spat_score - spat_score.min()) / (spat_score.max() - spat_score.min() + 1e-8)

    for strat in strategies:
        with torch.no_grad():
            model.reset()
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), 5e-3)
        optimizer.load_state_dict(optim_state_base)

        if strat == "Baseline SS-TPT (Entropy Selection)":
            score = 0.4 * suit_n + 0.6 * stab_n
            use_guided_select = False
        else:
            score = 0.35 * suit_n + 0.45 * stab_n + 0.20 * spat_n
            use_guided_select = True

        output_all = model(images)
        if use_guided_select:
            selected_idx = score.topk(3, largest=True).indices
            output = output_all[selected_idx]
        else:
            batch_entropy = -(output_all.softmax(1) * output_all.log_softmax(1)).sum(1)
            selected_idx = batch_entropy.topk(3, largest=False).indices
            output = output_all[selected_idx]

        weight_cons = F.softmax(score[selected_idx] / 0.25, dim=-1)
        loss = entropy_avg(output) + 1.0 * self_consistency_loss_with_rs_view(output, weight_cons, 'weighted')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tuned_outputs = model(images)
            weight_pred = F.softmax(score / 0.25, dim=-1)
            pred = torch.bmm(weight_pred.unsqueeze(0).unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)
            is_correct = (pred.argmax(dim=-1) == target).item()
            if is_correct:
                correct_counts[strat] += 1

    total += 1
    if (i + 1) % 10 == 0:
        print(f"Sample {i+1}/{N_SAMPLES}:")
        for s in strategies:
            print(f"  {s:<50}: {correct_counts[s] / total * 100:.2f}%")

print("\n--- FINAL STRATEGY COMPARISON (ImageNet-R) ---")
for s in strategies:
    print(f"{s:<50}: {correct_counts[s] / total * 100:.2f}%")


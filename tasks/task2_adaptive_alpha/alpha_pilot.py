"""Paired pilot: does a per-image alpha beat fixed 0.4 on the same views?

Tunes once at 0.4 and once at each candidate rule. Prints paired accuracy.
Does not change the SS-TPT training step, only which scalar alpha it receives.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

import torchattacks

from clip.custom_clip import get_coop
from data.cls_to_names import caltech101_classes
from data.datautils import AugMixAugmenter, build_dataset
from sstpt import (
    build_tiny_aug,
    compute_stability,
    select_confident_views,
    self_consistency_loss_with_rs_view,
    test_time_tuning,
)
from utils.tools import set_random_seed

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


GRID = (0.2, 0.3, 0.4, 0.5, 0.6)


def _pred(tuned, suit_n, stab_n, alpha, temp):
    score = alpha * suit_n + (1.0 - alpha) * stab_n
    weight = torch.softmax(score / temp, dim=0)
    mixed = torch.matmul(weight, tuned)
    return int(mixed.argmax().item())


def _kl(output_all, suit, stab, alpha, temp, k):
    output, idx = select_confident_views(output_all, k)
    ss = suit[idx]
    tt = stab[idx]
    sn = (ss - ss.min()) / (ss.max() - ss.min() + 1e-8)
    tn = (tt - tt.min()) / (tt.max() - tt.min() + 1e-8)
    score = alpha * sn + (1.0 - alpha) * tn
    weight = torch.softmax(score / temp, dim=0)
    return float(self_consistency_loss_with_rs_view(output, weight, "weighted").item())


def _choose_kl(kls):
    """Move off 0.4 only when consistency loss drops by at least 5%."""
    base = kls[0.4]
    cands = [a for a in GRID if a != 0.4 and kls[a] < base * 0.95]
    if not cands:
        return 0.4
    return min(cands, key=lambda a: kls[a])


def _choose_sens(pre_preds):
    """Move only when 0.3/0.4/0.5 disagree, toward the grid majority class."""
    if len({pre_preds[a] for a in (0.3, 0.4, 0.5)}) == 1:
        return 0.4
    counts = {}
    for a in GRID:
        counts[pre_preds[a]] = counts.get(pre_preds[a], 0) + 1
    majority = max(counts, key=counts.get)
    if counts[majority] < 3:
        return 0.4
    for a in (0.3, 0.5, 0.2, 0.6):
        if pre_preds[a] == majority and pre_preds[a] != pre_preds[0.4]:
            return a
    return 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/dse_413/ADL/data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.eps > 0 and args.steps <= 0:
        args.steps = 7
    pgd_alpha = args.eps / 4.0

    set_random_seed(args.seed)
    if hasattr(torchattacks, "config"):
        torchattacks.config.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    classnames = caltech101_classes
    model = get_coop("RN50", classnames, args.gpu, 4, "a_photo_of_a")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    model = model.cuda(args.gpu)
    model.eval()
    optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=5e-3)
    optim_state = deepcopy(optimizer.state_dict())

    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
    ])
    preprocess = transforms.Compose([transforms.ToTensor()])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=15, augmix=True)
    dataset = build_dataset("Caltech101", data_transform, args.data, mode="test")
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False))
    subset = torch.utils.data.Subset(dataset, pick.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    tune_args = argparse.Namespace(
        tta_steps=1, num_select=3, temp=0.25, cons_weight=1.0, ref_mode="weighted", gpu=args.gpu,
    )
    tiny = build_tiny_aug()
    if args.eps > 0:
        atk = torchattacks.PGD(model, eps=args.eps / 255, alpha=pgd_alpha / 255, steps=args.steps)

    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(), dim=1)
        logit_scale = model.logit_scale.exp()

    rules = ("fixed", "kl5", "sens")
    hit = {r: 0 for r in rules}
    moved = {r: 0 for r in rules}
    pre_hit = {a: 0 for a in GRID}
    pre_oracle = 0
    n_seen = 0
    kl_drop = []

    for batch_i, (images, target) in enumerate(loader):
        target = target.cuda(args.gpu, non_blocking=True)
        if args.eps > 0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv = atk(image, target)
            img_adv = transforms.ToPILImage()(adv.squeeze(0))
            images = data_transform(img_adv)
            images = [t.unsqueeze(0) for t in images]
        images = [t.cuda(args.gpu, non_blocking=True) for t in images]
        images = torch.cat(images, dim=0)

        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)
        with torch.no_grad():
            feats = model.image_encoder(model.normalize(images.type(model.dtype)))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (logit_scale * feats @ text_features.t()).float()
        suit_k = images.shape[0] - 1
        sim = torch.bmm(feats.unsqueeze(0), feats.unsqueeze(0).transpose(1, 2))
        dist = 2 - 2 * sim
        knn, _ = dist.topk(suit_k + 1, dim=-1, largest=False)
        suit = (1.0 / (knn[:, :, 1:].mean(-1) + 1e-8)).squeeze(0)
        stab = compute_stability(model, images, base_logits=logits, aug_builder=tiny)
        suit_n = (suit - suit.min()) / (suit.max() - suit.min() + 1e-8)
        stab_n = (stab - stab.min()) / (stab.max() - stab.min() + 1e-8)
        probs = torch.softmax(logits.float(), dim=-1)

        with torch.no_grad():
            kls = {a: _kl(logits, suit, stab, a, 0.25, 3) for a in GRID}
            pre = {a: _pred(probs, suit_n, stab_n, a, 0.25) for a in GRID}
        y = int(target.item())
        for a in GRID:
            pre_hit[a] += int(pre[a] == y)
        pre_oracle += int(any(pre[a] == y for a in GRID))
        kl_drop.append(min(kls.values()) / (kls[0.4] + 1e-12))

        chosen = {
            "fixed": 0.4,
            "kl5": _choose_kl(kls),
            "sens": _choose_sens(pre),
        }
        preds = {}
        cache = {}
        for rule, alpha in chosen.items():
            key = round(float(alpha), 4)
            if key not in cache:
                with torch.no_grad():
                    model.reset()
                optimizer.load_state_dict(optim_state)
                test_time_tuning(model, images, suit.detach(), stab.detach(), optimizer, None, tune_args, batch_i, alpha)
                with torch.no_grad():
                    tuned = model(images).float()
                cache[key] = _pred(tuned, suit_n, stab_n, alpha, 0.25)
            preds[rule] = cache[key]
            if abs(alpha - 0.4) > 1e-8:
                moved[rule] += 1
            hit[rule] += int(preds[rule] == y)
        n_seen += 1
        if n_seen % 50 == 0:
            print(
                f"n={n_seen} fixed={hit['fixed']/n_seen:.4f} "
                f"kl5={hit['kl5']/n_seen:.4f} sens={hit['sens']/n_seen:.4f} "
                f"moved_kl={moved['kl5']} moved_sens={moved['sens']}",
                flush=True,
            )

    print("--- summary ---", flush=True)
    print(f"eps={args.eps} n={n_seen}", flush=True)
    for rule in rules:
        print(f"{rule} acc={hit[rule]/n_seen:.5f} moved={moved[rule]}", flush=True)
    for a in GRID:
        print(f"pretune_a={a} acc={pre_hit[a]/n_seen:.5f}", flush=True)
    print(f"pretune_oracle={pre_oracle/n_seen:.5f}", flush=True)
    print(
        f"kl_ratio_min_over_04 mean={float(np.mean(kl_drop)):.4f} "
        f"p10={float(np.quantile(kl_drop, 0.1)):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

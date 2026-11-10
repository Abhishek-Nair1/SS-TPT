"""Paired test: average the predictions of prompts adapted at 0.3, 0.4, and 0.5.

Fixed alpha is a point estimate. The trace analysis showed that when some other
alpha is right and 0.4 is wrong, those other runs agree on the class. Averaging
a small prior around 0.4 keeps 0.4 when the neighbors disagree with each other,
and follows them when they agree.
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
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import caltech101_classes
from sstpt import build_tiny_aug, compute_stability, test_time_tuning
from utils.tools import set_random_seed

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

DEFAULT_ALPHAS = (0.3, 0.4, 0.5)


def _mixed(tuned, suit_n, stab_n, alpha, temp):
    score = alpha * suit_n + (1.0 - alpha) * stab_n
    weight = torch.softmax(score / temp, dim=0)
    return torch.matmul(weight, tuned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/dse_413/ADL/data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="0.3,0.4,0.5")
    args = parser.parse_args()
    alphas = tuple(sorted(float(x) for x in args.alphas.split(",")))
    if not any(abs(a - 0.4) < 1e-8 for a in alphas):
        raise SystemExit("alphas must include 0.4")
    if args.eps > 0 and args.steps <= 0:
        args.steps = 7
    pgd_alpha = args.eps / 4.0

    set_random_seed(args.seed)
    if hasattr(torchattacks, "config"):
        torchattacks.config.set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    model = get_coop("RN50", caltech101_classes, args.gpu, 4, "a_photo_of_a")
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
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    tune_args = argparse.Namespace(
        tta_steps=1, num_select=3, temp=0.25, cons_weight=1.0, ref_mode="weighted", gpu=args.gpu,
    )
    tiny = build_tiny_aug()
    if args.eps > 0:
        atk = torchattacks.PGD(model, eps=args.eps / 255, alpha=pgd_alpha / 255, steps=args.steps)

    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(), dim=1)
        logit_scale = model.logit_scale.exp()

    hit = {"fixed": 0, "hard": 0, "avg": 0}
    flipped = {"hard": 0, "avg": 0}
    gained = {"hard": 0, "avg": 0}
    lost = {"hard": 0, "avg": 0}
    neighbors_agree = 0
    n_seen = 0

    for batch_i, (images, target) in enumerate(loader):
        target = target.cuda(args.gpu, non_blocking=True)
        if args.eps > 0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv = atk(image, target)
            img_adv = transforms.ToPILImage()(adv.squeeze(0))
            images = [t.unsqueeze(0) for t in data_transform(img_adv)]
        images = torch.cat([t.cuda(args.gpu, non_blocking=True) for t in images], dim=0)

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

        mixed = {}
        labels = {}
        for alpha in alphas:
            with torch.no_grad():
                model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(
                model, images, suit.detach(), stab.detach(), optimizer, None, tune_args, batch_i, alpha
            )
            with torch.no_grad():
                tuned = model(images).float()
            mixed[alpha] = _mixed(tuned, suit_n, stab_n, alpha, 0.25)
            labels[alpha] = int(mixed[alpha].argmax().item())

        y = int(target.item())
        hard_counts = {}
        for alpha in alphas:
            hard_counts[labels[alpha]] = hard_counts.get(labels[alpha], 0) + 1
        hard_label, hard_n = max(hard_counts.items(), key=lambda kv: kv[1])
        hard = hard_label if hard_n >= 2 else labels[0.4]
        avg = int(torch.stack([mixed[a] for a in alphas]).mean(0).argmax().item())
        ends = (alphas[0], alphas[-1])
        if labels[ends[0]] == labels[ends[1]]:
            neighbors_agree += 1

        preds = {"fixed": labels[0.4], "hard": hard, "avg": avg}
        for name, pred in preds.items():
            hit[name] += int(pred == y)
        for name in ("hard", "avg"):
            if preds[name] != preds["fixed"]:
                flipped[name] += 1
                if preds[name] == y:
                    gained[name] += 1
                elif preds["fixed"] == y:
                    lost[name] += 1
        n_seen += 1
        if n_seen % 50 == 0:
            print(
                f"n={n_seen} fixed={hit['fixed']/n_seen:.4f} "
                f"hard={hit['hard']/n_seen:.4f} avg={hit['avg']/n_seen:.4f} "
                f"flip_hard={flipped['hard']} flip_avg={flipped['avg']}",
                flush=True,
            )

    print("--- summary ---", flush=True)
    print(f"eps={args.eps} n={n_seen} alphas={alphas} neighbors_agree={neighbors_agree}", flush=True)
    for name in ("fixed", "hard", "avg"):
        print(f"{name} acc={hit[name]/n_seen:.5f}", flush=True)
    for name in ("hard", "avg"):
        print(
            f"{name} flipped={flipped[name]} gained={gained[name]} lost={lost[name]} "
            f"net={gained[name]-lost[name]}",
            flush=True,
        )


if __name__ == "__main__":
    main()

import argparse
import csv
from datetime import datetime
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Ensure local modules (clip, data, utils, torchattacks) resolve properly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = 2

from clip.custom_clip import get_coop
from data.cls_to_names import *
from data.datautils import AugMixAugmenter, build_dataset
from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from utils.instance_alpha import (
    ALPHA_MODES,
    combine_ss,
    compute_instance_alpha,
    compute_instance_alpha_details,
)
from utils.spatial_metrics import (
    AddGaussianNoise,
    TinyAugmenterWithTransform,
    compute_pearson_spatial_similarity,
    min_max_normalize,
    combine_all_scores,
)
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed

import torchattacks

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div((p + eps).log(), (m + eps), reduction='none', log_target=False).sum(-1) +
                  F.kl_div((q + eps).log(), (m + eps), reduction='none', log_target=False).sum(-1))


def symmetric_kl(p, q, eps=1e-8):
    return 0.5 * (F.kl_div((p + eps).log(), (q + eps), reduction='none', log_target=False).sum(-1) +
                  F.kl_div((q + eps).log(), (p + eps), reduction='none', log_target=False).sum(-1))


def compute_stability_and_spatial_consistency(
    model,
    views,
    base_logits,
    aug_builder,
    K=1,
    divergence='js',
    use_spatial=False,
    spat_method='attn',
):
    model.eval()
    p = base_logits.softmax(dim=-1)

    if use_spatial:
        base_spatial_map = model.get_spatial_relevance_map(views, method=spat_method)
    else:
        base_spatial_map = None

    divergences = []
    spatial_sims = []

    for _ in range(K):
        aug_views, t_params = aug_builder(views)
        with torch.no_grad():
            q_logits = model(aug_views)
        q = q_logits.softmax(dim=-1)

        if divergence == 'js':
            d = js_divergence(p, q)
        elif divergence == 'sym':
            d = symmetric_kl(p, q)
        else:
            raise ValueError(f"Unsupported divergence: {divergence}")
        divergences.append(d)

        if use_spatial and base_spatial_map is not None:
            weak_spatial_map = model.get_spatial_relevance_map(aug_views, method=spat_method)
            aligned_base_map = aug_builder.align_spatial_map(base_spatial_map, t_params)
            sim = F.cosine_similarity(aligned_base_map.flatten(1), weak_spatial_map.flatten(1), dim=1, eps=1e-8)
            spatial_sims.append(sim)

    div = torch.stack(divergences, dim=0).mean(0)
    stab_score = 1.0 / (div + 1e-8)

    if use_spatial and len(spatial_sims) > 0:
        spat_score = torch.stack(spatial_sims, dim=0).mean(0)
    else:
        spat_score = None

    return stab_score, spat_score


def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()


def self_consistency_loss_with_rs_view(outputs, weight, ref_mode):
    eps = 1e-8
    probs = outputs.softmax(1).clamp_min(eps)

    if ref_mode == "weighted":
        weight = weight.unsqueeze(1)
        weighted_probs = weight * probs
        weighted_probs_sum = torch.sum(weighted_probs, dim=0, keepdim=True)
        loo_num = weighted_probs_sum - weighted_probs
        loo_den = (weight.sum(dim=0, keepdim=True) - weight).clamp_min(eps)
        ref_probs = (loo_num / loo_den).detach()
        kl = F.kl_div(probs.log(), ref_probs, reduction='batchmean', log_target=False)

    elif ref_mode == "weighted_self":
        weight = weight.unsqueeze(1)
        ref_prob = ((weight * probs).sum(dim=0, keepdim=True) / (weight.sum() + eps)).detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), reduction='batchmean', log_target=False)

    elif ref_mode == "best_score":
        ref_idx = torch.argmax(weight)
        ref_prob = probs[ref_idx].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), reduction='batchmean', log_target=False)

    elif ref_mode == "min_ent":
        ref_prob = probs[0].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), reduction='batchmean', log_target=False)

    elif ref_mode == "average":
        ref_prob = probs.mean(dim=0, keepdim=True).detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), reduction='batchmean', log_target=False)

    elif ref_mode == "random":
        ref_idx = torch.randint(probs.size(0), (1,), device=probs.device).item()
        ref_prob = probs[ref_idx].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), reduction='batchmean', log_target=False)
    else:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")

    return kl


def select_confident_views(outputs, scores, num_select, mode="entropy"):
    if mode == "entropy":
        entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        idx = torch.argsort(entropy, descending=False)[:num_select]
    elif mode == "score":
        idx = torch.argsort(scores, descending=True)[:num_select]
    elif mode == "pareto":
        entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        rank_ent = torch.argsort(torch.argsort(entropy, descending=False))
        rank_score = torch.argsort(torch.argsort(scores, descending=True))
        pareto_rank = rank_ent + rank_score
        idx = torch.argsort(pareto_rank, descending=False)[:num_select]
    else:
        idx = torch.arange(min(num_select, outputs.size(0)), device=outputs.device)
    return idx


def test_time_tuning(
    model,
    inputs,
    suit_score,
    stab_score,
    spat_score,
    optimizer,
    scaler,
    args,
    idx,
    alpha,
):
    selected_idx = None
    score_all, _ = combine_all_scores(
        suit_score=suit_score,
        stab_score=stab_score,
        spat_score=spat_score,
        alpha=alpha,
        spat_weight=args.spat_weight if getattr(args, 'use_spatial', False) else 0.0,
        combine_mode=getattr(args, 'combine_mode', 'gated'),
    )

    for j in range(args.tta_steps):
        output_all = model(inputs)

        if selected_idx is None:
            selected_idx = select_confident_views(
                output_all, score_all, args.num_select, mode=getattr(args, 'selection_mode', 'entropy')
            )
        output = output_all[selected_idx]

        loss = entropy_avg(output)

        if args.cons_weight > 0.0:
            weight = score_all[selected_idx]
            weight_pred = F.softmax(weight / args.temp, dim=-1)

            cons_loss = self_consistency_loss_with_rs_view(output, weight_pred, args.ref_mode)
            loss += args.cons_weight * cons_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return


def append_to_csv(csv_path, record):
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "timestamp",
        "method",
        "dataset",
        "epsilon",
        "attack_steps",
        "seed",
        "accuracy",
        "runtime",
        "alpha_mode",
        "spat_weight",
        "combine_mode",
        "samples",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description="Unified SS-TPT Framework (Baseline, Spatial, Adaptive Alpha, Combined)")
    parser.add_argument("data", metavar="DIR", help="path to dataset root")
    parser.add_argument("--test_sets", type=str, default="Caltech101")
    parser.add_argument("--dataset_mode", type=str, default="test")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="RN50")
    parser.add_argument("--resolution", default=224, type=int, help="CLIP image resolution")
    parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="number of views per sample")
    parser.add_argument("-p", "--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")

    parser.add_argument("--n_ctx", default=4, type=int, help="number of tunable context tokens")
    parser.add_argument("--ctx_init", default="a_photo_of_a", type=str, help="initial tunable prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output_results/sstpt")

    # Adversarial Attack Configuration
    parser.add_argument("--eps", default=0.0, type=float, help="attack perturbation budget (e.g. 1.0/255)")
    parser.add_argument("--alpha", default=0.0, type=float, help="attack step size (e.g. 0.25/255)")
    parser.add_argument("--steps", type=int, default=0, help="number of attack steps (0 = clean)")

    # TPT Optimization Hyperparameters
    parser.add_argument("--temp", type=float, default=0.25, help="softmax temperature for scores")
    parser.add_argument("--lr", "--learning-rate", default=5e-3, type=float, dest="lr", help="prompt learning rate")
    parser.add_argument("--num_select", default=3, type=int, help="number of selected views for adaptation")
    parser.add_argument("--tta_steps", default=1, type=int, help="number of test-time adaptation steps")
    parser.add_argument("--load_tecoa", type=str, default="", choices=["", "RN50-eps1", "ViT-B/32-eps4"])
    parser.add_argument("--cons_weight", type=float, default=1.0, help="consistency loss weight (lambda)")
    parser.add_argument("--ref_mode", type=str, default="weighted", choices=["weighted", "weighted_self", "best_score", "min_ent", "average", "random"])
    parser.add_argument("--suit_mode", type=str, default="density", choices=["sim", "density"])
    parser.add_argument("--ss_ratio", type=float, default=0.4, help="fixed trade-off ratio between suitability and stability")

    # Task 1: Cross-View Spatial Consistency Options
    parser.add_argument("--use_spatial", action="store_true", help="enable cross-view spatial consistency scoring")
    parser.add_argument("--spat_weight", type=float, default=0.10, help="weight gamma for spatial consistency")
    parser.add_argument("--spat_method", type=str, default="attn", choices=["attn", "gradcam"], help="spatial map extraction method")
    parser.add_argument("--combine_mode", type=str, default="gated", choices=["gated", "linear", "linear_norm"], help="view weighting combination mode")
    parser.add_argument("--tiny_k", type=int, default=1, help="number of weak augmentations for stability/spatial consistency")
    parser.add_argument("--selection_mode", type=str, default="entropy", choices=["entropy", "score", "pareto"], help="view selection strategy for TTA")

    # Task 2: Instance-Adaptive Stability-Suitability Balancing Options
    parser.add_argument("--alpha_mode", type=str, default="fixed", choices=list(ALPHA_MODES), help="instance-adaptive alpha mode")
    parser.add_argument("--alpha_grid", type=str, default="", help="optional comma-separated grid")

    # Person 3 Evaluation & Logging Harness Options
    parser.add_argument("--method_name", type=str, default="", help="explicit method name for reporting")
    parser.add_argument("--max_samples", type=int, default=-1, help="limit evaluation to N samples (-1 or 0 for full set)")
    parser.add_argument("--save_csv", type=str, default="", help="path to standardized CSV file to record experiment results")

    args = parser.parse_args()

    if not args.method_name:
        if not args.use_spatial and args.alpha_mode == "fixed":
            args.method_name = "SS-TPT baseline"
        elif args.use_spatial and args.alpha_mode == "fixed":
            args.method_name = "SS-TPT + Spatial"
        elif not args.use_spatial and args.alpha_mode != "fixed":
            args.method_name = f"SS-TPT + Adaptive α(x) ({args.alpha_mode})"
        else:
            args.method_name = f"SS-TPT + Spatial + Adaptive α(x) ({args.alpha_mode})"

    set_random_seed(args.seed)
    if hasattr(torchattacks, "config"):
        torchattacks.config.set_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        # cudnn.benchmark = True
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    if args.eps > 0 and args.steps <= 0:
        args.steps = 7
    if args.eps > 0 and args.alpha <= 0:
        args.alpha = args.eps / 4.0

    print(f"=== Unified SS-TPT Runner ===")
    print(f"Method: {args.method_name}")
    print(f"Dataset: {args.test_sets} | Epsilon: {args.eps} | Steps: {args.steps}")
    print(f"Alpha Mode: {args.alpha_mode} | Use Spatial: {args.use_spatial} | Combine Mode: {args.combine_mode}")
    print(f"Device: {device}")

    # Build data transform
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_transform = AugMixAugmenter(
        base_transform, preprocess, n_views=args.batch_size - 1, augmix=(len(args.test_sets) > 1)
    )

    val_dataset = build_dataset(args.test_sets, data_transform, args.data, mode=args.dataset_mode)
    print(f"Number of test samples in dataset: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    dset = args.test_sets
    if len(dset) > 1:
        classnames = eval(f"{dset.lower()}_classes")
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval(f"imagenet_{dset.lower()}_mask")
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    # Initialize CLIP model
    model = get_coop(args.arch, classnames, device, args.n_ctx, args.ctx_init)
    if args.load_tecoa != "":
        model = load_model_weight(args.load_tecoa, model, device)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    model = model.to(device)
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    # Prepare weak augmentation builder
    tiny_aug = TinyAugmenterWithTransform()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    out_file_path = os.path.join(args.output_dir, "output.txt")
    out_file = open(out_file_path, "w")
    out_file.write(print_args(args) + "\n")
    out_file.flush()

    trace_file_path = os.path.join(args.output_dir, "alpha_trace.jsonl")
    trace_file = open(trace_file_path, "w")

    # Evaluation loop
    start_time = time.time()
    tpt_acc1 = AverageMeter("Acc@1", ":6.2f")
    total_samples = len(val_loader) if args.max_samples <= 0 else min(args.max_samples, len(val_loader))

    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = F.normalize(text_features, dim=1)
        logit_scale = model.logit_scale.exp()

    if args.eps > 0.0:
        atk = torchattacks.PGD(model, eps=args.eps / 255.0, alpha=args.alpha / 255.0, steps=args.steps)

    for i, (images, target) in enumerate(val_loader):
        if args.max_samples > 0 and i >= args.max_samples:
            break

        target = target.to(device, non_blocking=True)

        # Apply Adversarial Attack if configured
        if args.eps > 0.0:
            image = images[0].to(device, non_blocking=True)
            adv_image = atk(image, target)
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=True)
            images = torch.cat(images, dim=0)
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=True)

        # Reset tunable prompt for independent instance-level adaptation
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        # Extract features and initial prediction
        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()

        # Compute suitability scores
        suit_k = images.shape[0] - 1
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        if args.suit_mode == "sim":
            sorted_vals, _ = sim_matrix.sort(dim=-1, descending=True)
            top_k_mean = sorted_vals[..., 1 : suit_k + 1].mean(dim=-1)
            suit_score = top_k_mean.squeeze(0)
        elif args.suit_mode == "density":
            dist = 2 - 2 * sim_matrix
            knn_dists, _ = dist.topk(suit_k + 1, dim=-1, largest=False)
            suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8)
            suit_score = suit_score.squeeze(0)
        else:
            raise ValueError(f"Unsupported suit_mode: {args.suit_mode}")

        # Compute stability and spatial consistency
        stab_score, spat_score = compute_stability_and_spatial_consistency(
            model=model,
            views=images,
            base_logits=clip_outputs,
            aug_builder=tiny_aug,
            K=args.tiny_k,
            divergence="js",
            use_spatial=args.use_spatial,
            spat_method=args.spat_method,
        )

        # Instance-Adaptive Stability-Suitability Trade-off alpha(x)
        suit_n_all = min_max_normalize(suit_score)
        stab_n_all = min_max_normalize(stab_score)

        if args.alpha_mode != "fixed":
            alpha, r_suit, r_stab = compute_instance_alpha_details(
                suit_n_all,
                stab_n_all,
                mode=args.alpha_mode,
                fixed_alpha=args.ss_ratio,
                temp=args.temp,
                topk=args.num_select,
                suit_raw=suit_score,
                stab_raw=stab_score,
            )
        else:
            alpha = torch.tensor(args.ss_ratio, device=device)
            r_suit = torch.tensor(0.0, device=device)
            r_stab = torch.tensor(0.0, device=device)

        # Test-Time Prompt Adaptation
        test_time_tuning(
            model=model,
            inputs=images,
            suit_score=suit_score.detach(),
            stab_score=stab_score.detach(),
            spat_score=spat_score.detach() if spat_score is not None else None,
            optimizer=optimizer,
            scaler=None,
            args=args,
            idx=i,
            alpha=alpha,
        )

        # Post-adaptation forward pass
        with torch.no_grad():
            tuned_outputs = model(images)

        # Final unified view weighting
        final_score, details = combine_all_scores(
            suit_score=suit_score,
            stab_score=stab_score,
            spat_score=spat_score,
            alpha=alpha,
            spat_weight=args.spat_weight if args.use_spatial else 0.0,
            combine_mode=args.combine_mode,
        )
        weight_pred = F.softmax(final_score / args.temp, dim=-1)

        # Prediction aggregation
        tta_output = torch.bmm(weight_pred.unsqueeze(0).unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        acc1 = accuracy(tta_output, target, topk=(1,))[0]
        tpt_acc1.update(acc1.item(), 1)

        pred_label = int(tta_output.argmax().item())
        target_label = int(target.flatten()[0].item())

        trace = {
            "index": int(i),
            "target": target_label,
            "pred": pred_label,
            "correct": bool(pred_label == target_label),
            "alpha": round(float(alpha.detach().cpu().item()), 6),
            "r_suit": round(float(r_suit.detach().cpu().item()), 6),
            "r_stab": round(float(r_stab.detach().cpu().item()), 6),
            "spat_weight": float(args.spat_weight) if args.use_spatial else 0.0,
            "combine_mode": args.combine_mode,
            "mode": args.alpha_mode,
            "eps": float(args.eps),
        }
        trace_file.write(json.dumps(trace) + "\n")
        trace_file.flush()

        if (i + 1) % args.print_freq == 0 or (i + 1) == total_samples:
            log_str = f"iter:{i+1}/{total_samples}, acc1={tpt_acc1.avg:.2f}%"
            print(log_str)
            out_file.write(log_str + "\n")
            out_file.flush()

    total_runtime = time.time() - start_time
    final_acc = tpt_acc1.avg
    summary_str = (
        f"\n==========================================\n"
        f"Completed: {args.method_name}\n"
        f"Samples Evaluated: {total_samples}\n"
        f"Top-1 Accuracy: {final_acc:.2f}%\n"
        f"Total Runtime: {total_runtime:.2f}s ({total_runtime / max(1, total_samples):.4f} s/sample)\n"
        f"==========================================\n"
    )
    print(summary_str)
    out_file.write(summary_str)
    out_file.close()
    trace_file.close()

    # Record to standardized CSV if requested
    if args.save_csv:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "method": args.method_name,
            "dataset": args.test_sets,
            "epsilon": float(args.eps),
            "attack_steps": int(args.steps),
            "seed": int(args.seed),
            "accuracy": round(float(final_acc), 4),
            "runtime": round(float(total_runtime), 2),
            "alpha_mode": args.alpha_mode,
            "spat_weight": float(args.spat_weight) if args.use_spatial else 0.0,
            "combine_mode": args.combine_mode,
            "samples": int(total_samples),
        }
        append_to_csv(args.save_csv, record)
        print(f"Appended results to: {args.save_csv}")

    return final_acc


if __name__ == "__main__":
    main()

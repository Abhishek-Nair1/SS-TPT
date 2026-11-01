import os
import sys
from pathlib import Path

# Self-contained path resolution for clip, data, utils, torchattacks
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for p in [str(CURRENT_DIR), str(PARENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np

import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import random

import torchattacks
import matplotlib.pyplot as plt
import pandas as pd


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
        
def visualize_all_metrics(suit_n, stab_n, combined_score, idx, save_path=None):
    import matplotlib.pyplot as plt

    def to_np(x):
        try:
            return x.detach().float().cpu().numpy()
        except Exception:
            return np.asarray(x, dtype=float)

    s = to_np(suit_n)
    t = to_np(stab_n)
    c = to_np(combined_score)

    style = {
        "font.family": "Times New Roman",
        "font.size": 20,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12.5, 8))
        x = np.arange(1, len(s) + 1)

        ax.plot(x, s, label='Suitability (z)', linewidth=5, marker='o', markersize=20, linestyle='-')
        ax.plot(x, t, label='Stability (z)',   linewidth=5, marker='x', markersize=20, linestyle='--')
        ax.plot(x, c, label='SS Score',        linewidth=5, marker='s', markersize=20, linestyle='-.')

        ax.set_title(f"Suitability, Stability, and SS Scores (iter {idx})", pad=12, fontweight='bold')
        ax.set_xlabel("Sample Index", fontsize=40, fontweight='bold', family="Times New Roman")
        ax.set_ylabel("Score", fontsize=45, fontweight='bold', family="Times New Roman")

        ax.set_xticks(x)
        all_vals = np.concatenate([s, t, c])
        vr = max(all_vals.max() - all_vals.min(), 1e-8)
        pad = max(0.6, vr * 0.05)
        ax.set_ylim([float(all_vals.min()) - pad, float(all_vals.max()) + pad])

        ax.tick_params(axis='both', labelsize=40)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        leg = ax.legend(loc='best', frameon=True)
        leg.get_frame().set_linewidth(1.0)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.02):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    def forward(self, x):
        sigma = torch.empty(1, device=x.device).uniform_(self.sigma_min,
                                                         self.sigma_max)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0, 1)

def build_tiny_aug(
    rotation=5.0,
    translate=0.04,
    scale_var=0.08, 
    brightness=0.08,
    contrast=0.08,
    saturation=0.08,
    hue=0.02,
    blur_sigma=(0.5, 1.2),
    noise_sigma=0.02,
):
    affine = transforms.RandomAffine(
        degrees=rotation,
        translate=(translate, translate),
        scale=(1.0 - scale_var, 1.0 + scale_var),
        shear=0.0,
        interpolation=InterpolationMode.BILINEAR
    )

    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    tiny_aug = transforms.Compose([
        transforms.RandomApply([affine], p=1.0), 
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=blur_sigma)], p=0.5),
        transforms.RandomApply([AddGaussianNoise(0.0, noise_sigma)], p=0.5),
    ])
    return tiny_aug

def select_confident_views(logits, k):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    # We select at least two confident samples for the consistency loss.    
    # k = min(batch_entropy.size(0), max(int(batch_entropy.size(0) * p), 3))
    k = min(batch_entropy.size()[0], max(2, k))
    idx = batch_entropy.topk(k, largest=False).indices
    return logits[idx], idx

def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div((p+eps).log(), (m+eps), reduction='none', log_target=False).sum(-1) +
                  F.kl_div((q+eps).log(), (m+eps), reduction='none', log_target=False).sum(-1))

def symmetric_kl(p, q, eps=1e-8):
    return 0.5 * (F.kl_div((p+eps).log(), (q+eps), reduction='none', log_target=False).sum(-1) +
            F.kl_div((q+eps).log(), (p+eps), reduction='none', log_target=False).sum(-1))
    
class TinyAugmenterWithTransform(torch.nn.Module):
    def __init__(
        self,
        rotation=5.0,
        translate=0.04,
        scale_var=0.08, 
        brightness=0.08,
        contrast=0.08,
        saturation=0.08,
        hue=0.02,
        blur_sigma=(0.5, 1.2),
        noise_sigma=0.02,
    ):
        super().__init__()
        self.rotation = rotation
        self.translate = translate
        self.scale_var = scale_var
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.blur_sigma = blur_sigma
        self.noise_sigma = noise_sigma

    def forward(self, x):
        B, C, H, W = x.shape
        angle = float(torch.empty(1).uniform_(-self.rotation, self.rotation).item())
        max_dx = float(self.translate * W)
        max_dy = float(self.translate * H)
        tx = float(torch.empty(1).uniform_(-max_dx, max_dx).item())
        ty = float(torch.empty(1).uniform_(-max_dy, max_dy).item())
        scale = float(torch.empty(1).uniform_(1.0 - self.scale_var, 1.0 + self.scale_var).item())
        
        x_aug = TF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR
        )
        
        if random.random() < 0.5:
            x_aug = self.color_jitter(x_aug)
        if random.random() < 0.5:
            sigma = float(torch.empty(1).uniform_(self.blur_sigma[0], self.blur_sigma[1]).item())
            x_aug = TF.gaussian_blur(x_aug, kernel_size=[5, 5], sigma=[sigma, sigma])
        if random.random() < 0.5:
            noise = torch.randn_like(x_aug) * (random.random() * self.noise_sigma)
            x_aug = (x_aug + noise).clamp(0, 1)

        transform_params = {
            'angle': angle,
            'translate': [tx, ty],
            'scale': scale,
            'shear': [0.0, 0.0]
        }
        return x_aug, transform_params

    def align_spatial_map(self, spatial_map, transform_params):
        return TF.affine(
            spatial_map,
            angle=transform_params['angle'],
            translate=transform_params['translate'],
            scale=transform_params['scale'],
            shear=transform_params['shear'],
            interpolation=InterpolationMode.BILINEAR
        )

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

        if use_spatial:
            weak_spatial_map = model.get_spatial_relevance_map(aug_views, method=spat_method)
            aligned_base_map = aug_builder.align_spatial_map(base_spatial_map, t_params)
            crop_margin = 16
            aligned_crop = aligned_base_map[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
            weak_crop = weak_spatial_map[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
            # Pearson correlation (zero-mean centering) removes positive DC offset
            u = aligned_crop.flatten(1)
            v = weak_crop.flatten(1)
            u_c = u - u.mean(dim=1, keepdim=True)
            v_c = v - v.mean(dim=1, keepdim=True)
            sim = F.cosine_similarity(u_c, v_c, dim=1, eps=1e-8)
            spatial_sims.append(sim)

    div = torch.stack(divergences, dim=0).mean(0)  # [B]
    stab_score = 1 / (div + 1e-8)  

    if use_spatial:
        spat_score = torch.stack(spatial_sims, dim=0).mean(0) # [B]
    else:
        spat_score = None

    return stab_score, spat_score

def compute_cross_view_spatial_consistency(attn):
    spatial_centered = attn - attn.mean(dim=-1, keepdim=True)
    spatial_norm = torch.nn.functional.normalize(spatial_centered, p=2, dim=-1)
    S_spat = torch.mm(spatial_norm, spatial_norm.t()) # [B, B]
    B = attn.size(0)
    consensus = (S_spat.sum(dim=1) - 1.0) / (B - 1.0)
    base_sim = S_spat[:, 0]
    spat_score = 0.5 * consensus + 0.5 * base_sim
    return spat_score

def combine_scores(suit_score, stab_score, spat_score, args):
    suit_n = (suit_score - suit_score.min()) / (suit_score.max() - suit_score.min() + 1e-8)
    stab_n = (stab_score - stab_score.min()) / (stab_score.max() - stab_score.min() + 1e-8)
    ss_base = args.ss_ratio * suit_n + (1.0 - args.ss_ratio) * stab_n
    if getattr(args, 'spat_weight', 0.0) > 0.0 and spat_score is not None:
        spat_n = (spat_score - spat_score.min()) / (spat_score.max() - spat_score.min() + 1e-8)
        combine_mode = getattr(args, 'combine_mode', 'gated')
        if combine_mode == 'linear':
            score = (1.0 - args.spat_weight) * ss_base + args.spat_weight * spat_n
        else:
            score = ss_base * ((1.0 - args.spat_weight) + args.spat_weight * spat_n)
    else:
        score = ss_base
    return score

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def self_consistency_loss_with_rs_view(outputs, weight, ref_mode):
    eps = 1e-8
    probs = outputs.softmax(1).clamp_min(eps) # [B,C]

    if ref_mode == "weighted":
        weight = weight.unsqueeze(1)
        weighted_probs = weight * probs # [B,C]
        weighted_probs_sum = torch.sum(weighted_probs, dim=0, keepdim=True) # [1,C]
        # Leave one out
        loo_num = weighted_probs_sum - weighted_probs # [B,C]
        loo_den = (weight.sum(dim=0, keepdim=True) - weight).clamp_min(eps)  # [B,1]
        ref_probs = (loo_num / loo_den).detach()
        kl = F.kl_div(probs.log(), ref_probs, reduction='batchmean', log_target=False)

    elif ref_mode == "weighted_self":
        weight = weight.unsqueeze(1)
        ref_prob = ((weight * probs).sum(dim=0, keepdim=True) / (weight.sum() + eps)).detach() 
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
            reduction='batchmean', log_target=False)

    elif ref_mode == "best_score":
        ref_idx = torch.argmax(weight)
        ref_prob = probs[ref_idx].detach()  
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), 
            reduction='batchmean', log_target=False)
        
    elif ref_mode == "min_ent":
        ref_prob = probs[0].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs), 
            reduction='batchmean', log_target=False)
        
    elif ref_mode == "average":
        ref_prob = probs.mean(dim=0, keepdim=False).detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
                      reduction='batchmean', log_target=False)

    elif ref_mode == "random":
        ref_idx = torch.randint(probs.size(0), (1,), device=probs.device).item()
        ref_prob = probs[ref_idx].detach()
        kl = F.kl_div(probs.log(), ref_prob.expand_as(probs),
                      reduction='batchmean', log_target=False)
    else:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")
    
    return kl

def test_time_tuning(model, inputs, suit_score, stab_score, spat_score, optimizer, scaler, args, idx):
    selected_idx = None
    score_all = combine_scores(suit_score, stab_score, spat_score, args)

    for j in range(args.tta_steps):

        output_all = model(inputs) 

        if selected_idx is not None:
            output = output_all[selected_idx]
        else:
            if getattr(args, 'selection_mode', 'entropy') == 'pareto':
                cand = torch.topk(score_all, 8, largest=True).indices
                cand_ent = -(output_all[cand].softmax(1) * output_all[cand].log_softmax(1)).sum(1)
                best_cand = torch.topk(cand_ent, args.num_select, largest=False).indices
                selected_idx = cand[best_cand]
                output = output_all[selected_idx]
                weight_cons = torch.nn.functional.softmax(score_all[selected_idx]/args.temp, dim=-1)
            elif getattr(args, 'selection_mode', 'entropy') == 'score':
                selected_idx = score_all.topk(args.num_select, largest=True).indices
                output = output_all[selected_idx]
                weight_cons = torch.nn.functional.softmax(score_all[selected_idx]/args.temp, dim=-1)
            else:
                output, selected_idx = select_confident_views(output_all, args.num_select)
                weight_cons = torch.nn.functional.softmax(score_all[selected_idx]/args.temp, dim=-1)
        
        ent_loss = entropy_avg(output)
        
        if args.cons_weight > 0:
            assert output.size(0) > 1
            kl_loss = self_consistency_loss_with_rs_view(output, weight_cons, args.ref_mode)
            loss = ent_loss + args.cons_weight  * kl_loss
        else:
            loss = ent_loss
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def main():
    t0 = time.time()
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    if hasattr(torchattacks, "config"):
        torchattacks.config.set_seed(args.seed)
    
    if 'alpha' not in vars(args) or args.alpha == 0.0:
        args.alpha = args.eps / 4.0

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        if args.gpu is not None:
            print(f"GPU {args.gpu} requested but CUDA is unavailable; running on CPU instead.")

    safe_arch = args.arch.replace('/', '_')
    out_name = 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps)+'_batchsize_'+str(args.batch_size)
    if getattr(args, 'spat_weight', 0.0) > 0.0:
        out_name += '_spat_' + str(args.spat_weight) + '_' + str(args.spat_method)
        if getattr(args, 'tiny_k', 1) > 1:
            out_name += '_k' + str(args.tiny_k)
    args.output_dir = os.path.join(args.output_dir, safe_arch, args.test_sets, out_name)

    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    print(f"Using device: {device}")

    # model
    dset = args.test_sets
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    model = get_coop(args.arch, classnames, str(device), args.n_ctx, args.ctx_init)
    model_state = None

    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': '../pretrain/tecoa/weights/backbone/rn50_eps1.pth.tar',
            'ViT-B/32-eps4': '../pretrain/tecoa/weights/backbone/vitb32_eps4.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        print('load robust vision encoder')

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    else:
        print('using CPU, this will be slow')
    model = model.to(device)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    # cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 1

        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(dset))
        
        t1 = time.time()
        elapsed_ds = t1 - t0
        log_msg = f"[{dset}] Loading time: {elapsed_ds:.2f} seconds\n"
        print(log_msg.strip())
        args.out_file.write(log_msg)
        args.out_file.flush()
        
        result = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, device)
        del val_dataset, val_loader
        if args.eps <= 0:
            print_log = "=> Acc. on testset [{}]: SSTPT Clean Acc @1 {}".format(dset, result)
            save_log = {'sstpt_clean_acc': result}
        else:
            print_log = "=> Acc. on testset [{}]: SSTPT Adv Acc @1 {} ".format(dset, result)
            save_log = {'sstpt_adv_acc': result}
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        elapsed_ds = time.time() - t1
        log_msg = f"[{dset}] Evaluation time: {elapsed_ds:.2f} seconds\n"
        print(log_msg.strip())
        args.out_file.write(log_msg)
        args.out_file.flush()

        # torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))
        

def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, device):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    tpt1 = AverageMeter('SSTPTAcc@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, tpt1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        
    end = time.time()

    tiny_aug = TinyAugmenterWithTransform()
    
    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = F.normalize(text_features, dim=1)
        logit_scale = model.logit_scale.exp()
    
    for i, (images, target) in enumerate(val_loader):
        if getattr(args, 'max_samples', -1) > 0 and i >= args.max_samples:
            break
        target = target.to(device, non_blocking=True)

        if args.eps > 0.0:
            image = images[0].to(device, non_blocking=True)
            adv_image = atk(image, target)
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=True)
            image = images
        
        images = torch.cat(images, dim=0)

        # reset model
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        # with torch.no_grad():
        #     clip_features, _, _ = model.forward_features(images)
        #     clip_outputs = model(images)

        with torch.no_grad():
            clip_features = model.image_encoder(model.normalize(images.type(model.dtype)))
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_outputs = logit_scale * clip_features @ text_features.t()

        use_spatial = (getattr(args, 'spat_weight', 0.0) > 0.0)
        spat_method = getattr(args, 'spat_method', 'attn')

        stab_score, spat_score = compute_stability_and_spatial_consistency(
            model, images, base_logits=clip_outputs,
            aug_builder=tiny_aug,
            K=getattr(args, 'tiny_k', 1),
            use_spatial=use_spatial,
            spat_method=spat_method
        )

        suit_k = images.shape[0] - 1
        sim_matrix = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1)) # [1, B, B]
        if args.suit_mode == "sim":
            sorted_vals, _ = sim_matrix.sort(dim=-1, descending=True)
            top_k_mean = sorted_vals[..., 1:suit_k+1].mean(dim=-1) 
            suit_score = top_k_mean.squeeze(0)
        elif args.suit_mode == "density":
            dist = 2 - 2 * sim_matrix  # [1, B, B]
            knn_dists, _ = dist.topk(suit_k+1, dim=-1, largest=False)
            suit_score = 1.0 / (knn_dists[:, :, 1:].mean(-1) + 1e-8) # [1, B]
            suit_score = suit_score.squeeze(0) # [B]
        else: 
            raise ValueError(f"Unsupported suit_mode: {args.suit_mode}")

        assert args.tta_steps > 0
        test_time_tuning(model, images, suit_score.detach(), stab_score.detach(),
                         spat_score.detach() if spat_score is not None else None,
                         optimizer, scaler, args, i)
        with torch.no_grad():
            tuned_outputs = model(images)
            
        score = combine_scores(suit_score, stab_score, spat_score, args)
        weight_pred = torch.nn.functional.softmax(score/args.temp, dim=-1)
        tta_output = torch.bmm(weight_pred.unsqueeze(0).unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))
       
        tpt1.update(tpt_acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        eff_total = args.max_samples if getattr(args, 'max_samples', -1) > 0 else len(val_loader)
        if (i+1) % args.print_freq == 0 or (i+1) == eff_total or (i+1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, sstpt_acc1={}'.format(i, len(val_loader), tpt1.avg)
            else:
                print_log = 'iter:{}/{}, sstpt_adv1={}'.format(i, len(val_loader), tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    progress.display_summary()

    return tpt1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    DEFAULT_DATA_DIR = str(CURRENT_DIR / 'data') if (CURRENT_DIR / 'data').exists() else '/home/abhishek/workspace/SS-TPT/data'
    parser.add_argument('data', metavar='DIR', nargs='?', default=DEFAULT_DATA_DIR, help=f'path to dataset root (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/sstpt')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--temp', type=float, default=0.25,
                        help='softmax temperature')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--num_select', default=3, type=int, help='number of selected views')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps4'])
    
    parser.add_argument('--cons_weight', type=float, default=1.0,
                        help='weight for consistency loss (λ)')
    parser.add_argument('--ref_mode', type=str, default='weighted',
                    choices=['weighted', 'weighted_self', 'best_score', 'min_ent', 'average', 'random'],
                    help='strategy to select reference predictions for consistency loss computation')
    parser.add_argument('--suit_mode', type=str, default='density', choices=['sim', 'density'],
                        help="method for computing suitability weights: 'sim' (cosine similarity) or 'density' (inverse mean kNN distance)")
    parser.add_argument('--ss_ratio', type=float, default=0.4,
                        help='weighting ratio to combine suitability and stability scores')
    parser.add_argument('--tiny_k', type=int, default=1,
                        help='number of weak augmentations for stability/spatial consistency (default: 1)')
    parser.add_argument('--spat_weight', type=float, default=0.10, # Best setting
                        help='weighting ratio for spatial relevance consistency score (0.0 means original SS-TPT)')
    parser.add_argument('--selection_mode', type=str, default='entropy', choices=['entropy', 'score', 'pareto'], help='view selection strategy: entropy or score')
    parser.add_argument('--combine_mode', type=str, default='gated', choices=['gated', 'linear'], help='score combination method: gated (multiplicative gate) or linear (additive)')
    parser.add_argument('--spat_method', type=str, default='attn', choices=['attn', 'gradcam', 'crossview'],
                        help="method to extract spatial relevance map ('attn' or 'gradcam')")
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='maximum number of test samples to evaluate (-1 for full dataset)')
    main()

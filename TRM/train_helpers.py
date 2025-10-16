import math
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
from torch import autocast

from papers_e2e.TRM import get_targets


# -----------------------------
# Training step (skeleton)
# -----------------------------

class CallModelTRM:
    """Wrapper so your existing get_targets() can stay unchanged.
    It returns a single velocity tensor just like before.
    - Teacher uses EMA if available (use_ema=True) and by default returns v_main
    to avoid circular training on the refiner. You can flip to v_ref if desired.
    - Temperature is set to 0.0 for teachers (deterministic weighting),
    but you can change it via ctor.
    """

    def __init__(self, model: nn.Module, ema_model: Optional[nn.Module] = None, teacher_use_refined: bool = False,
                 teacher_temperature: float = 0.0):
        self.model = model
        self.ema_model = ema_model
        self.teacher_use_refined = teacher_use_refined
        self.teacher_temperature = teacher_temperature

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, k: torch.Tensor, y: torch.Tensor,
                 use_ema: bool = False) -> torch.Tensor:
        net = self.ema_model if (use_ema and (self.ema_model is not None)) else self.model
        # Teacher always in eval-ish mode; no top1 gating to keep grads well-shaped when used elsewhere
        out = net(x, t, k, y, train=False, temperature=self.teacher_temperature, top1=False)
        v = out['v_ref'] if self.teacher_use_refined else out['v_main']
        return v


def trm_losses(out: Dict[str, torch.Tensor], v_target: torch.Tensor,
               lambda_flow: float = 1.0,
               lambda_quality: float = 0.5,
               lambda_align: float = 0.25,
               lambda_diversity: float = 0.05,
               align_alpha: float = 8.0,
               diversity_tau: float = 0.2,
               eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Compute TRM losses given model outputs and FM/Shortcut target v_target.
    out contains: v_main (N,C,H,W), v_ref (N,C,H,W), v_props (N,K,C,H,W), weights (N,K)
    """
    v_main = out['v_main']
    v_ref  = out['v_ref']
    v_props = out['v_props']
    w = out['weights']  # (N,K)

    # 1) Main flow loss
    L_flow_main = F.mse_loss(v_main, v_target)

    # 2) Quality loss on refined velocity
    L_quality = F.mse_loss(v_ref, v_target)

    # 3) Soft alignment: encourage at least one proposal to match v_ref
    #    d_k = ||v_k - v_ref||^2 (averaged over pixels)
    N, K, C, H, W = v_props.shape
    diffs = (v_props - v_ref.unsqueeze(1)).pow(2).mean(dim=(2,3,4))  # (N,K)
    w_softmin = torch.softmax(-align_alpha * diffs, dim=-1)           # (N,K)
    L_align = (w_softmin * diffs).sum(dim=-1).mean()

    # 4) Diversity: penalize high cosine similarity between different proposals
    #    We compute pairwise cosine over flattened maps, then hinge above tau.
    v_flat = F.normalize(v_props.view(N, K, -1), dim=-1)  # (N,K,CHW)
    # pairwise cosine matrix per sample
    cos = torch.einsum('nkd,nld->nkl', v_flat, v_flat)  # (N,K,K)
    mask = torch.triu(torch.ones(K, K, device=cos.device, dtype=torch.bool), diagonal=1)
    pair_cos = cos[:, mask]  # (N, K*(K-1)/2)
    L_div = F.relu(pair_cos - diversity_tau).mean() if pair_cos.numel() > 0 else v_main.sum()*0

    # Optional logging helpers
    with torch.no_grad():
        weight_entropy = -(w.clamp_min(1e-6) * (w.clamp_min(1e-6)).log()).sum(dim=-1).mean()
        avg_spread = diffs.mean()

    # Weighted sum
    loss = (lambda_flow * L_flow_main +
            lambda_quality * L_quality +
            lambda_align * L_align +
            lambda_diversity * L_div)

    return {
        'loss': loss,
        'L_flow_main': L_flow_main.detach(),
        'L_quality': L_quality.detach(),
        'L_align': L_align.detach(),
        'L_diversity': L_div.detach(),
        'weight_entropy': weight_entropy,
        'avg_prop_ref_dist': avg_spread,
    }


# --- one training iteration (DDP-safe, AMP-safe) ---
def train_step(step, images, labels, cfg, gen,
               model,
               ema_model=None, amp_dtype=torch.bfloat16):

    # 0) Teacher for get_targets(): returns a single velocity like before (v_main by default)
    call_teacher = CallModelTRM(
        model=model,
        ema_model=ema_model,
        teacher_use_refined=False,    # start with main head as teacher to avoid circularity
        teacher_temperature=0.0
    )

    # 1) Build targets exactly as before (Heun teacher + optional CFG + FM split)
    with torch.no_grad():
        x_t, v_t, t, k, y, _ = get_targets(cfg, gen, images, labels, call_teacher, step)

    # 2) Forward pass through TRM model (returns dict)
    trm_temp = getattr(cfg.model_cfg, "trm_temperature", 5.0)  # soft blend sharpness
    with autocast('cuda', dtype=amp_dtype):
        out = model(x_t, t, k, y, train=True, temperature=trm_temp, top1=False)

        # 3) Losses (same target v_t for both FM + shortcut parts)
        #    You can ramp some weights by step if you like
        def cosine_ramp(cur, start, end, target):
            if cur <= start: return 0.0
            if cur >= end:   return float(target)
            p = (cur - start) / max(1, (end - start))
            return float(0.5 * (1 - math.cos(math.pi * p)) * target)

        lam_flow     = getattr(cfg.model_cfg, "lambda_flow", 1.0)                 # constant
        lam_quality  = cosine_ramp(step,
                                   start=getattr(cfg.model_cfg, "quality_ramp_start",   10_000),
                                   end=getattr(cfg.model_cfg,   "quality_ramp_end",     60_000),
                                   target=getattr(cfg.model_cfg, "lambda_quality",      0.5))
        lam_align    = cosine_ramp(step,
                                   start=getattr(cfg.model_cfg, "align_ramp_start",     40_000),
                                   end=getattr(cfg.model_cfg,   "align_ramp_end",       90_000),
                                   target=getattr(cfg.model_cfg, "lambda_align",        0.25))
        lam_div      = getattr(cfg.model_cfg, "lambda_diversity", 0.05)

        align_alpha  = getattr(cfg.model_cfg, "align_alpha", 8.0)
        diversity_tau= getattr(cfg.model_cfg, "diversity_tau", 0.2)

        losses = trm_losses(
            out, v_t,
            lambda_flow=lam_flow,
            lambda_quality=lam_quality,
            lambda_align=lam_align,
            lambda_diversity=lam_div,
            align_alpha=align_alpha,
            diversity_tau=diversity_tau
        )

    # 6) Return scalars for logging
    scalars = {k: (v.detach().item() if v.numel() == 1 else v.detach().mean().item())
               for k, v in losses.items() if torch.is_tensor(v)}
    scalars["loss"] = float(losses["loss"].item())
    scalars["lambda_quality_now"] = lam_quality
    scalars["lambda_align_now"] = lam_align

    return losses, scalars



# -----------------------------
# Inference helper (sampler uses this to get v)
# -----------------------------
@torch.no_grad()
def predict_v_for_sampler(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, k: torch.Tensor, y: torch.Tensor,
                          use_top1: bool = False, temperature: float = 5.0) -> torch.Tensor:
    """Return the velocity used by your sampler. Default: refined blend.
    If you want sharper but potentially unstable choices across time, set use_top1=True.
    """
    out = model(x_t, t, k, y, train=False, temperature=temperature, top1=use_top1)
    return out['v_ref']


# -----------------------------
# Example config knobs to add
# -----------------------------
# cfg.model_cfg.trm_temperature = 5.0     # soft weighting sharpness during train/infer
# cfg.model_cfg.lambda_flow = 1.0
# cfg.model_cfg.lambda_quality = 0.5
# cfg.model_cfg.lambda_align = 0.25
# cfg.model_cfg.lambda_diversity = 0.05
# cfg.model_cfg.align_alpha = 8.0
# cfg.model_cfg.diversity_tau = 0.2
# cfg.runtime_cfg.use_trm_top1_inference = False

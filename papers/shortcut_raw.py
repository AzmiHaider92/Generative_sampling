import math
import torch

EPS = 1e-5  # endpoint epsilon for the linear path

"""
Implementation of the paper: One Step Diffusion via Shortcut Models
https://arxiv.org/abs/2410.12557
"""


@torch.no_grad()
def get_targets(cfg, gen, images, labels, call_model_fn, step, force_t: float = -1.0, force_dt: float = -1.0):
    """
    PyTorch port of get_targets (JAX). Returns:
      x_t, v_t, t, dt_base, labels_dropped, info
    """
    device = images.device
    B = images.shape[0]
    denoise_timesteps = int(cfg.model_cfg.denoise_timesteps)
    bootstrap_every = cfg.model_cfg.bootstrap_every
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    # ---------- 1) Sample dt (bootstrap) ----------
    bootstrap_batchsize = B // bootstrap_every
    log2_sections = int(math.log2(denoise_timesteps))  # e.g., 128 -> 7

    if cfg.model_cfg.bootstrap_dt_bias == 0:
        # dt_base in {log2_sections-1, ..., 0}
        levels = torch.arange(log2_sections - 1, -1, -1, device=device, dtype=torch.long)
        reps = max(1, bootstrap_batchsize // max(1, log2_sections))
        dt_base = levels.repeat_interleave(reps)
        if dt_base.numel() < bootstrap_batchsize:
            pad = torch.zeros(bootstrap_batchsize - dt_base.numel(), device=device, dtype=torch.long)
            dt_base = torch.cat([dt_base, pad], dim=0)
        dt_base = dt_base[:bootstrap_batchsize]  # (B_b,)
        num_dt_cfg = bootstrap_batchsize // max(1, log2_sections)
    else:
        # biased mix (like JAX version)
        levels = torch.arange(log2_sections - 1, -1, -1, device=device, dtype=torch.long)
        # the JAX code uses (bootstrap_batchsize // 2) // log2_sections for repeats of shortened levels
        short_levels = torch.arange(log2_sections - 3, -1, -1, device=device, dtype=torch.long)  # log2_sections-2 terms
        reps = max(1, (bootstrap_batchsize // 2) // max(1, log2_sections))
        dt_base = short_levels.repeat_interleave(reps)  # shrink
        # then append 1s and 0s quarters
        part = bootstrap_batchsize // 4
        dt_base = torch.cat([dt_base,
                             torch.ones(part, device=device, dtype=torch.long),
                             torch.zeros(part, device=device, dtype=torch.long)], dim=0)
        if dt_base.numel() < bootstrap_batchsize:
            pad = torch.zeros(bootstrap_batchsize - dt_base.numel(), device=device, dtype=torch.long)
            dt_base = torch.cat([dt_base, pad], dim=0)
        dt_base = dt_base[:bootstrap_batchsize]
        num_dt_cfg = max(1, (bootstrap_batchsize // 2) // max(1, log2_sections))

    if force_dt != -1:
        # force_dt is interpreted like JAX: overrides dt_base directly (exponent values)
        dt_base = torch.full((bootstrap_batchsize,), int(force_dt), device=device, dtype=torch.long)

    # dt = 1 / 2^{dt_base}   e.g., [1, 1/2, 1/4, ...]
    dt = 1.0 / (2.0 ** dt_base.to(torch.float32))                    # (B_b,)
    dt_bootstrap = dt / 2.0                                          # (B_b,)

    # ---------- 2) Sample t (bootstrap) ----------
    dt_sections = 2 ** dt_base                                       # (B_b,) int
    # randint per-example in [0, dt_sections[i])
    # Do it by sampling in max range then mod:
    max_section = int(dt_sections.max().item()) if dt_sections.numel() > 0 else 1
    t_raw = torch.randint(low=0, high=max_section, size=(bootstrap_batchsize,), device=device, generator=gen)

    t_mod = (t_raw % torch.clamp(dt_sections, min=1))                # (B_b,)
    t = t_mod.to(torch.float32) / torch.clamp(dt_sections.to(torch.float32), min=1.0)  # in [0,1)
    if force_t != -1:
        t = torch.full_like(t, float(force_t))

    t_full = t.view(bootstrap_batchsize, *([1] * (images.ndim - 1)))  # [B_b, 1, 1, 1]

    # ---------- 3) Bootstrap targets ----------
    x_1 = images[:bootstrap_batchsize]                                # clean
    x_0 = torch.randn(x_1.shape, dtype=x_1.dtype, device=x_1.device, generator=gen)

    x_t = (1.0 - (1.0 - EPS) * t_full) * x_0 + t_full * x_1           # convex combo
    bst_labels = labels[:bootstrap_batchsize]

    if not cfg.model_cfg.bootstrap_cfg:
        v_b1 = call_model_fn(x_t, t, dt_bootstrap, bst_labels, train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap.view(-1, *([1] * (images.ndim - 1))) * v_b1
        x_t2 = torch.clamp(x_t2, -4.0, 4.0)
        v_b2 = call_model_fn(x_t2, t2, dt_bootstrap, bst_labels, train=False)
        v_target = 0.5 * (v_b1 + v_b2)
    else:
        # CFG duplication
        x_t_extra = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
        t_extra = torch.cat([t, t[:num_dt_cfg]], dim=0)
        dt_extra = torch.cat([dt_bootstrap, dt_bootstrap[:num_dt_cfg]], dim=0)
        labels_extra = torch.cat([bst_labels, torch.full((num_dt_cfg,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_extra, labels_extra, train=False)
        v_b1_cond = v_b1_raw[:x_1.shape[0]]
        v_b1_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b1_uncond + cfg_scale * (v_b1_cond[:num_dt_cfg] - v_b1_uncond)
        v_b1 = torch.cat([v_cfg, v_b1_cond[num_dt_cfg:]], dim=0)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap.view(-1, *([1] * (images.ndim - 1))) * v_b1
        x_t2 = torch.clamp(x_t2, -4.0, 4.0)

        x_t2_extra = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
        t2_extra = torch.cat([t2, t2[:num_dt_cfg]], dim=0)

        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)

        v_target = 0.5 * (v_b1 + v_b2)

    v_target = torch.clamp(v_target, -4.0, 4.0)
    bst_v  = v_target
    bst_dt = dt.clone()                    # int64
    bst_t  = t.clone()                          # float32
    bst_xt = x_t
    bst_l  = bst_labels

    # ---------- 4) Flow-Matching targets ----------
    # label dropout (CFG training trick)
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().to(dtype=torch.long)
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    # sample t in {0..denoise_timesteps-1} then / denoise_timesteps
    t_flow = torch.randint(low=0, high=denoise_timesteps, size=(B,), device=device, generator=gen).to(torch.float32) / float(denoise_timesteps)

    if force_t != -1:
        t_flow = torch.full_like(t_flow, float(force_t))
    t_flow_full = t_flow.view(B, *([1] * (images.ndim - 1)))

    x1_flow = images
    x0_flow = torch.randn(x1_flow.shape, dtype=x1_flow.dtype, device=x1_flow.device, generator=gen)

    x_t_flow = (1.0 - (1.0 - EPS) * t_flow_full) * x0_flow + t_flow_full * x1_flow
    v_t_flow = x1_flow - (1.0 - EPS) * x0_flow

    dt_flow_int = 1 / denoise_timesteps #int(math.log2(denoise_timesteps))
    dt_flow = torch.full((B,), dt_flow_int, device=device, dtype=torch.long)

    # ---------- 5) Merge Flow + Bootstrap ----------
    bst_size = B // bootstrap_every
    bst_size_data = B - bst_size

    x_t_out = torch.cat([bst_xt, x_t_flow[:bst_size_data]], dim=0)
    t_out = torch.cat([bst_t, t_flow[:bst_size_data]], dim=0)
    dt_out = torch.cat([bst_dt, dt_flow[:bst_size_data]], dim=0)          # int64
    v_t_out = torch.cat([bst_v, v_t_flow[:bst_size_data]], dim=0)
    labels_out = torch.cat([bst_l, labels_dropped[:bst_size_data]], dim=0)

    return x_t_out, v_t_out, t_out, dt_out, labels_out, None

import math
import torch

EPS = 1e-5  # endpoint epsilon for the linear path

"""
Implementation of the paper: One Step Diffusion via Shortcut Models
https://arxiv.org/abs/2410.12557
"""


def sample_dt_t_warp(
    twarper,
        cfg,
    device,
        batch_size,
    gen,
    force_level: float = -1,   # interpreted as a *bin index* override for warper
    force_t: float = -1,
):
    """
    Warped sampler using the learned time_warper schedule.

    Returns (same shapes/types as your uniform sampler):
      dt, dt_half                   (B,) float
      k_code, k_teacher_code        (B,) float   (for warper we expose a *continuous* code = -log2(dt))
      t                              (B,) float   (midpoint; ensures t +/- dt/2 in [0,1])
      num_dt_cfg                     int
    """


    # Constants
    T = int(cfg.model_cfg.denoise_timesteps)
    K = int(math.log2(T))

    # === Section 1: choose bootstrap subset size ===
    bootstrap_every = int(cfg.model_cfg.bootstrap_every)
    bootstrap_size = min(batch_size, batch_size // max(1, bootstrap_every))

    B = int(bootstrap_size)
    if B == 0:
        z = torch.empty(0, device=device)
        return z, z, z, z, z, 0

    if K <= 0:
        raise ValueError("denoise_timesteps must be > 0")

    # ---- Build non-uniform grid t_grid of length (K+1), descending noisy->clean ----
    # IMPORTANT: keep autograd enabled so the warper learns.
    # We call the warper's *forward* on a uniform u-grid to get the mapped times.
    u_grid = torch.linspace(0.0, 1.0, K + 1, device=device)  # ascending u
    tw = twarper.module if hasattr(twarper, "module") else twarper
    t_grid, _ = tw(u_grid)
    t_grid = t_grid.clamp(1e-6, 1 - 1e-6)  # numeric safety

    # Interval widths (positive)
    dt_bins = (t_grid[:-1] - t_grid[1:]).abs()  # shape (K,)

    # ---- Pick an interval index k for each sample ----
    if force_level != -1:
        # interpret force_level as a *bin index* request for warper mode
        k = torch.full((B,), int(max(0, min(K-1, int(force_level)))), device=device, dtype=torch.long)
    else:
        k = torch.randint(0, K, (B,), device=device, generator=gen)  # uniform over bins

    # Gather hi/lo times and widths
    t_lo = t_grid[k]          # (B,)
    t_hi = t_grid[k + 1]      # (B,)
    dt = (t_hi - t_lo)        # (B,)
    dt_half = 0.5 * dt

    # ---- Choose the midpoint t (so that interval [t - dt/2, t + dt/2] stays in [0,1]) ----
    if force_t != -1:
        t_mid = torch.full((B,), float(force_t), device=device)
    else:
        # sample uniformly such that t_mid ∈ [dt/2, 1 - dt/2]
        low = dt_half
        high = 1.0 - dt_half
        # torch.rand(B)*(high-low)+low, but high/low are per-sample
        t_mid = torch.rand(B, device=device, generator=gen) * (high - low) + low

    # ---- Codes (continuous) ----
    # Your uniform version uses a "continuous level code" s with dt = 2^{-s}.
    # For warper, dt is arbitrary; we expose a compatible code: s := -log2(dt).
    # The teacher code is s+1 (clamped) to mimic your behavior.
    eps = 1e-12
    k_code = (-torch.log2(dt + eps)).to(t_mid.dtype)              # (B,)
    # Max level corresponds to the *smallest* bin in this grid
    s_max = (-torch.log2(dt_bins.clamp_min(eps).min() + eps)).detach()
    k_teacher_code = torch.minimum(k_code + 1.0, s_max.expand_as(k_code))

    # Heuristic duplication factor: mirror the logic (bins ~ K)
    num_dt_cfg = B // max(1, K)

    return dt, dt_half, k_code, k_teacher_code, t_mid, int(num_dt_cfg)


@torch.no_grad()
def get_targets3_5(cfg, gen, images, labels, call_model,
                dt_half, k_code, k_teacher_code, t, num_dt_cfg,
                force_t: float = -1):
    """
    Supports two (k,t) sampling schemes:
      - dt_mode='bins'          : discrete dyadic levels + grid-aligned t
      - dt_mode='uniform_log'   : log-uniform levels (continuous), t~U[0,1-dt/2]
      - dt_mode='uniform_linear': linear-uniform dt, t~U[0,1-dt/2]
    Returns:
      x_t_out, v_t_out, t_out, k_out, labels_out, info
    """
    device = images.device
    B = images.shape[0]

    # Constants
    T = int(cfg.model_cfg.denoise_timesteps)
    K = int(math.log2(T))

    # === Section 1: choose bootstrap subset size ===
    bootstrap_every = int(cfg.model_cfg.bootstrap_every)
    bootstrap_size = min(B, B // max(1, bootstrap_every))

    t_full = t.view(-1, 1, 1, 1)

    # === Section 3: Bootstrap targets (Heun teacher, optional CFG) ===
    if bootstrap_size > 0:
        x1 = images[:bootstrap_size]
        x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)

        # linear path with epsilon
        x_t = (1.0 - (1.0 - EPS) * t_full) * x0 + t_full * x1
        b_labels = labels[:bootstrap_size].to(dtype=torch.long)

        use_ema = True # use the ema to produce teacher targets
        if not cfg.model_cfg.bootstrap_cfg:
            v_b1 = call_model(x_t, t, k_teacher_code, b_labels, use_ema=use_ema)
            x_t2 = torch.clamp(x_t + dt_half.view(-1, 1, 1, 1) * v_b1, -4, 4)
            v_b2 = call_model(x_t2, t + dt_half, k_teacher_code, b_labels, use_ema=use_ema)
            v_target = 0.5 * (v_b1 + v_b2)
        else:
            num_dt_cfg = min(num_dt_cfg, bootstrap_size)
            x_t_ext = torch.cat([x_t, x_t[:num_dt_cfg]], 0)
            t_ext = torch.cat([t, t[:num_dt_cfg]], 0)
            k_ext = torch.cat([k_teacher_code, k_teacher_code[:num_dt_cfg]], 0)
            labels_ext = torch.cat(
                [b_labels, torch.full((num_dt_cfg,), cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0,
                                      device=device, dtype=torch.long)], 0
            )

            v_raw = call_model(x_t_ext, t_ext, k_ext, labels_ext, use_ema=use_ema)
            v_cond, v_uncond = v_raw[:bootstrap_size], v_raw[bootstrap_size:]
            v_cfg = v_uncond + float(cfg.model_cfg.cfg_scale) * (v_cond[:num_dt_cfg] - v_uncond)
            v_b1 = torch.cat([v_cfg, v_cond[num_dt_cfg:]], 0)

            x_t2 = torch.clamp(x_t + dt_half.view(-1, 1, 1, 1) * v_b1, -4, 4)
            x_t2_ext = torch.cat([x_t2, x_t2[:num_dt_cfg]], 0)
            t2_ext = torch.cat([t + dt_half, (t + dt_half)[:num_dt_cfg]], 0)

            v2_raw = call_model(x_t2_ext, t2_ext, k_ext, labels_ext, use_ema=use_ema)
            v2_cond, v2_uncond = v2_raw[:bootstrap_size], v2_raw[bootstrap_size:]
            v2_cfg = v2_uncond + float(cfg.model_cfg.cfg_scale) * (v2_cond[:num_dt_cfg] - v2_uncond)
            v_b2 = torch.cat([v2_cfg, v2_cond[num_dt_cfg:]], 0)

            v_target = 0.5 * (v_b1 + v_b2)

        v_target = torch.clamp(v_target, -4, 4)
        bst_v, bst_k, bst_t, bst_xt, bst_l = v_target, k_code, t, x_t, b_labels
    else:
        bst_v = images.new_empty((0,) + images.shape[1:])
        bst_k = torch.empty(0, device=device)     # float level code
        bst_t = torch.empty(0, device=device)
        bst_xt = images.new_empty((0,) + images.shape[1:])
        bst_l = torch.empty(0, dtype=torch.long, device=device)
        v_b1 = v_b2 = images.new_zeros(1)

    # === Section 4: Flow-matching targets (global) ===
    rest = B - bootstrap_size
    if rest > 0:
        t_flow = torch.randint(0, T, (rest,), generator=gen, device=device).float()
        t_flow = t_flow / float(T)
        if force_t != -1:
            t_flow = torch.full_like(t_flow, float(force_t))
        t_flow_full = t_flow.view(rest, 1, 1, 1)

        x1_flow = images[:rest]
        x0_flow = torch.randn(x1_flow.shape, dtype=x1_flow.dtype, device=x1_flow.device, generator=gen)

        # linear path & FM target with epsilon
        x_t_flow = (1.0 - (1.0 - EPS) * t_flow_full) * x0_flow + t_flow_full * x1_flow
        v_t_flow = x1_flow - (1.0 - EPS) * x0_flow

        K_float = float(K)
        k_flow = torch.full((rest,), K_float, device=device)  # sentinel level code

        # label dropout (CFG training trick)
        p = float(cfg.model_cfg.class_dropout_prob)
        drop = torch.bernoulli(torch.full((rest,), p, device=device)).bool()
        labels_flow = labels[:rest].clone().to(dtype=torch.long)
        labels_flow[drop] = cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0

        x_t_out = torch.cat([bst_xt, x_t_flow], 0)
        v_t_out = torch.cat([bst_v, v_t_flow], 0)
        t_out = torch.cat([bst_t, t_flow], 0)
        k_out = torch.cat([bst_k, k_flow], 0)    # float level code for the model
        labels_out = torch.cat([bst_l, labels_flow], 0)
    else:
        x_t_out, v_t_out, t_out, k_out, labels_out = bst_xt, bst_v, bst_t, bst_k, bst_l

    return x_t_out, v_t_out, t_out, k_out, labels_out, None



def get_targets(cfg, gen, batch_images, batch_labels, call_teacher_model, step, twarper):
    dt, dt_half, k_code, k_teacher_code, t, num_dt_cfg = sample_dt_t_warp(twarper, cfg, batch_images.device,
                                                                          batch_images.shape[0], gen)
    # create targets
    x_t, v_t, t_vec, k_vec, labels_eff, info = get_targets3_5(cfg, gen, batch_images, batch_labels,
                                                           call_teacher_model,
                                                           dt_half, k_code, k_teacher_code, t, num_dt_cfg,
                                                           step)
    return x_t, v_t, t_vec, k_vec, labels_eff, info
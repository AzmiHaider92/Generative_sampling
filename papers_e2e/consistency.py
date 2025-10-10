import math
import torch

EPS = 1e-5

"""
Implementation of the paper: Consistency Models (2023)
https://arxiv.org/abs/2303.01469
"""


@torch.no_grad()
def get_targets(cfg, gen, images, labels, call_model, step, force_t: float = -1, force_dt: float = -1):
    """
      - Picks a dyadic level k based on the current train_step
      - Samples grid-aligned t on that level
      - Forms x_t and x_{t+dt}
      - Uses EMA model at (t+dt, k) to build a one-shot 'shortcut' target

    Returns:
      x_t:        (B, C, H, W)
      v_target:   (B, C, H, W)
      t:          (B,)
      k_level:    (B,)  float level code (the per-sample k; constant here)
      labels:     (B,)  unchanged
      info:       dict  with diagnostics
    """
    device = images.device
    B = images.shape[0]

    # ----- 1) Sample k (level) from training step schedule -----
    T = int(cfg.model_cfg.denoise_timesteps)      # power of two
    K = int(math.log2(T))                         # max level
    # Partition training into K phases → k ∈ {0,...,K-1}
    # floor(step / (max_steps / K))
    phase = (cfg.runtime_cfg.max_steps / max(K, 1))
    k_int = int(math.floor(step / max(phase, 1e-8)))
    k_int = max(0, min(K - 1, k_int))            # clamp for safety

    if force_dt != -1:
        k_int = int(force_dt)

    k_level = torch.full((B,), float(k_int), device=device)  # (B,)

    dt = 2.0 ** (-k_int)           # scalar step size for this batch
    dt_sections = 2 ** k_int       # number of bins on [0,1]

    # ----- 2) Sample t on the k-grid -----
    # t ∈ {0, 1/2^k, 2/2^k, ..., (2^k-1)/2^k}
    t_bins = torch.randint(low=0, high=dt_sections, size=(B,), generator=gen, device=device)
    t = t_bins.float() / float(dt_sections)
    if force_t != -1:
        t = torch.full_like(t, float(force_t))

    t2 = t + float(dt)  # can equal 1.0 for the last bin (ok with EPS path)
    t_full  = t.view(B, 1, 1, 1)
    t2_full = t2.view(B, 1, 1, 1)

    # ----- 3) Construct states on the linear path -----
    x1 = images
    x0 = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen)

    x_t  = (1.0 - (1.0 - EPS) * t_full)  * x0 + t_full  * x1
    x_t2 = (1.0 - (1.0 - EPS) * t2_full) * x0 + t2_full * x1

    # ----- 4) One EMA forward at (t+dt, k) and shortcut target -----
    # v_b2 = f_ema(x_{t+dt}, t+dt, k, y)
    v_b2 = call_model(x_t2, t2, k_level, labels, use_ema=True)

    # pred_x1 = x_{t+dt} + (1 - (t+dt)) * v_b2
    pred_x1 = x_t2 + (1.0 - t2_full) * v_b2

    # v_target = (pred_x1 - x_t) / (1 - t)
    v_target = (pred_x1 - x_t) / (1.0 - t_full)

    return x_t, v_target, t, k_level, labels, None


import math
import torch

EPS = 1e-5  # endpoint epsilon for the linear path

"""
Implementation of the paper: One Step Diffusion via Shortcut Models
https://arxiv.org/abs/2410.12557
"""


@torch.no_grad()
def _build_k_bins(bootstrap_size: int, K: int, device, bias: int):
    """
    Original paper's *binned* scheme for level indices.
    Returns:
      k_long: (B_b,) int64 with levels k ∈ {0 .. K-1}
      num_dt_cfg: int for CFG duplication in bootstrap
    """
    if bootstrap_size <= 0:
        return torch.empty(0, dtype=torch.long, device=device), 0

    if bias == 0:
        levels = torch.arange(K - 1, -1, -1, device=device, dtype=torch.long)  # [K-1 .. 0]
        reps = max(1, bootstrap_size // max(1, K))
        k_long = levels.repeat_interleave(reps)
        if k_long.numel() < bootstrap_size:
            k_long = torch.cat([k_long,
                                torch.zeros(bootstrap_size - k_long.numel(), device=device, dtype=torch.long)], 0)
        k_long = k_long[:bootstrap_size]
        num_dt_cfg = bootstrap_size // max(1, K)
    else:
        # Biased: chunk over [K-1 .. 2], then a quarter ones, a quarter zeros, then pad
        upper = torch.arange(K - 1, 1, -1, device=device, dtype=torch.long)  # [K-1 .. 2]
        reps = max(1, (bootstrap_size // 2) // max(1, K))
        a = upper.repeat_interleave(reps)
        b = torch.ones(bootstrap_size // 4, device=device, dtype=torch.long)
        c = torch.zeros(bootstrap_size // 4, device=device, dtype=torch.long)
        k_long = torch.cat([a, b, c], 0)
        if k_long.numel() < bootstrap_size:
            k_long = torch.cat([k_long,
                                torch.zeros(bootstrap_size - k_long.numel(), device=device, dtype=torch.long)], 0)
        k_long = k_long[:bootstrap_size]
        num_dt_cfg = max(1, (bootstrap_size // 2) // max(1, K))

    return k_long, int(num_dt_cfg)


@torch.no_grad()
def sample_dt_t_bins(bootstrap_size: int, denoise_timesteps: int, device, gen, bias: int,
                     force_level: float = -1, force_t: float = -1):
    """
    Original 'bins' method (discrete dyadic levels).
    Returns:
      dt, dt_half                   (B_b,) float
      k_code, k_teacher_code        (B_b,) float  (level codes for student/teacher)
      t                              (B_b,) float
      num_dt_cfg                     int

    # example: denoise_timesteps = 128, bootstrap_size=8
    # K=log(128) = 7, levels k -{6,5,4,3,2,1,0,0},
    # for each level k => dt=2^-k and t ∈ {0,dt,2dt,…,1−dt}
    # for example: ki=6 => dt = 1/64 and t ∈ {1/64, 2/64, 3/64, ...., 63/64}
    #
    """
    K = int(math.log2(int(denoise_timesteps)))
    k_long, num_dt_cfg = _build_k_bins(bootstrap_size, K, device, bias)

    if bootstrap_size == 0:
        z = torch.empty(0, device=device)
        return z, z, z, z, z, 0

    if force_level != -1:
        k_long = torch.full_like(k_long, int(force_level))

    # step sizes
    dt = torch.pow(2.0, -k_long.float())
    dt_half = 0.5 * dt

    # level codes for model embedders
    k_code = k_long.float()                  # student level (k)
    k_teacher_code = k_code + 1.0            # teacher level (k+1)  (≤ K in bins)

    # grid-aligned t over {0, dt, 2dt, ..., 1-dt}
    dt_sections = torch.pow(2.0, k_long.float())
    u = torch.rand(bootstrap_size, device=device, generator=gen)
    t_bins = torch.floor(u * dt_sections).to(torch.long)
    t_bins = torch.minimum(t_bins, dt_sections.to(torch.long) - 1)
    t = t_bins.float() / dt_sections
    if force_t != -1:
        t = torch.full_like(t, float(force_t))

    return dt, dt_half, k_code, k_teacher_code, t, num_dt_cfg


@torch.no_grad()
def sample_dt_t_uniform(bootstrap_size: int, denoise_timesteps: int, device, gen,
                        mode: str = "log", dt_min: float | None = None,
                        force_level: float = -1, force_t: float = -1):
    """
    Uniform samplers (continuous levels).
      mode="log":   s ~ U[0,K], dt=2^{-s}  (scale-balanced; recommended)
      mode="linear": dt ~ U[dt_min, 1]
    Returns:
      dt, dt_half                   (B_b,) float
      k_code, k_teacher_code        (B_b,) float  (continuous level codes)
      t                              (B_b,) float  (U[0, 1-dt/2])
      num_dt_cfg                     int
    """
    if bootstrap_size == 0:
        z = torch.empty(0, device=device)
        return z, z, z, z, z, 0

    K = float(math.log2(int(denoise_timesteps)))

    if mode == "linear":
        if dt_min is None:
            dt_min = 2.0 ** (-K)  # as fine as the original schedule
        dt = torch.empty(bootstrap_size, device=device).uniform_(dt_min, 1.0, generator=gen)
        k_code = -torch.log2(dt)  # continuous level code
    elif mode == "log":
        s = torch.empty(bootstrap_size, device=device).uniform_(0.0, K, generator=gen)  # s in [0, K]
        k_code = s.clone()          # level code = s
        dt = torch.pow(2.0, -k_code)
    else:
        raise ValueError("mode must be 'log' or 'linear'")

    if force_level != -1:
        k_code = torch.full_like(k_code, float(force_level))
        dt = torch.pow(2.0, -k_code)

    dt_half = 0.5 * dt

    # Ensure t + dt/2 <= 1
    t = torch.rand(bootstrap_size, device=device, generator=gen) * (1.0 - dt_half)
    if force_t != -1:
        t = torch.full_like(t, float(force_t))

    # Teacher level = k+1, but clamp to K to keep embedder range consistent
    K_tensor = torch.tensor(K, device=device)
    k_teacher_code = torch.minimum(k_code + 1.0, K_tensor)

    # heuristic: same duplication budget as bins
    num_dt_cfg = bootstrap_size // max(1, int(K))
    return dt, dt_half, k_code, k_teacher_code, t, int(num_dt_cfg)


@torch.no_grad()
def get_targets(cfg, gen, images, labels, call_model, step, force_t: float = -1, force_dt: float = -1):
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

    # === Sections 1–2: sample (k,t) & step sizes (two schemes) ===
    dt_mode = getattr(cfg.model_cfg, "dt_mode", "bins")
    if dt_mode == "bins":
        dt, dt_half, k_code, k_teacher_code, t, num_dt_cfg = sample_dt_t_bins(
            bootstrap_size=bootstrap_size,
            denoise_timesteps=T,
            device=device, gen=gen,
            bias=int(cfg.model_cfg.bootstrap_dt_bias),
            force_level=force_dt, force_t=force_t
        )
    elif dt_mode == "uniform_log":
        dt, dt_half, k_code, k_teacher_code, t, num_dt_cfg = sample_dt_t_uniform(
            bootstrap_size=bootstrap_size,
            denoise_timesteps=T,
            device=device, gen=gen,
            mode="log",
            force_level=force_dt, force_t=force_t
        )
    elif dt_mode == "uniform_linear":
        dt, dt_half, k_code, k_teacher_code, t, num_dt_cfg = sample_dt_t_uniform(
            bootstrap_size=bootstrap_size,
            denoise_timesteps=T,
            device=device, gen=gen,
            mode="linear", dt_min=None,
            force_level=force_dt, force_t=force_t
        )
    else:
        raise ValueError(f"Unknown dt_mode: {dt_mode}")

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
                [b_labels, torch.full((num_dt_cfg,), int(cfg.runtime_cfg.num_classes),
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
        labels_flow[drop] = int(cfg.runtime_cfg.num_classes)

        x_t_out = torch.cat([bst_xt, x_t_flow], 0)
        v_t_out = torch.cat([bst_v, v_t_flow], 0)
        t_out = torch.cat([bst_t, t_flow], 0)
        k_out = torch.cat([bst_k, k_flow], 0)    # float level code for the model
        labels_out = torch.cat([bst_l, labels_flow], 0)
    else:
        x_t_out, v_t_out, t_out, k_out, labels_out = bst_xt, bst_v, bst_t, bst_k, bst_l

    return x_t_out, v_t_out, t_out, k_out, labels_out, None

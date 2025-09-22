
import math
import torch


def _build_dt_base(bootstrap_size: int, log2_sections: int, device, bias: int):
    """
    Return dt_base of length = bootstrap_size.
    bias == 0  -> cycle [L-1, ..., 0] as needed, or truncate if bootstrap_size < L
    bias != 0  -> first half uses [L-3, ..., 0] cycled, then ~1/4 of 1s and ~1/4 of 0s
    """
    if bootstrap_size <= 0:
        return torch.empty(0, device=device, dtype=torch.float32)

    L = int(log2_sections)
    if L <= 0:
        return torch.zeros(bootstrap_size, device=device, dtype=torch.float32)

    if bias == 0:
        # dt_list = [L-1, ..., 0]
        dt_list = torch.arange(L - 1, -1, -1, device=device, dtype=torch.float32)
        if bootstrap_size <= L:
            return dt_list[:bootstrap_size]
        reps = math.ceil(bootstrap_size / L)
        return dt_list.repeat(reps)[:bootstrap_size]
    else:
        # head uses [L-3, ..., 0] (avoid the two coarsest), cycled to ~half of the slice
        start = max(L - 3, 0)
        head_list = torch.arange(start, -1, -1, device=device, dtype=torch.float32)
        k = bootstrap_size // 2
        if head_list.numel() == 0:
            head = torch.zeros(k, device=device, dtype=torch.float32)
        else:
            reps = math.ceil(k / head_list.numel())
            head = head_list.repeat(reps)[:k]

        # tail: ~1/4 ones then the rest zeros
        rem = bootstrap_size - k
        q = rem // 2
        tail = torch.empty(rem, device=device, dtype=torch.float32)
        if q > 0:
            tail[:q] = 1.0
        if rem - q > 0:
            tail[q:] = 0.0
        return torch.cat([head, tail], 0)


@torch.no_grad()
def get_targets(cfg, gen, call_model, images, labels, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = images.shape[0]  # per-rank batch

    # --- bootstrap sizes from *local* batch ---
    bootstrap_every = int(cfg.model_cfg.bootstrap_every)
    bootstrap_size = B // max(1, bootstrap_every)  # e.g., B=8, every=4 -> 2
    bootstrap_size = min(bootstrap_size, B)

    log2_sections = int(math.log2(int(cfg.model_cfg.denoise_timesteps)))

    # build dt_base for bootstrap slice (length = bootstrap_size)
    dt_base = _build_dt_base(
        bootstrap_size=bootstrap_size,
        log2_sections=log2_sections,
        device=device,
        bias=int(cfg.model_cfg.bootstrap_dt_bias),
    )
    if force_dt != -1:
        dt_base = torch.full((bootstrap_size,), float(force_dt), device=device)

    dt = 1.0 / (2.0 ** dt_base)
    dt_base_bootstrap = dt_base + 1.0
    dt_bootstrap = dt / 2.0

    # sample t for bootstrap slice (length = bootstrap_size)
    if bootstrap_size == 0:
        t = dt_base.new_zeros((0,))
    else:
        dt_sections = (2.0 ** dt_base)                       # [bootstrap_size]
        u = torch.rand(bootstrap_size, device=device, generator=gen)
        t_bins = torch.floor(u * dt_sections).to(torch.int64)
        t_bins = torch.minimum(t_bins, dt_sections.to(torch.int64) - 1)
        t = t_bins.float() / dt_sections
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(-1, 1, 1, 1)

    # bootstrap pairs (length = bootstrap_size)
    x1 = images[:bootstrap_size]
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)
    #x0 = torch.randn_like(x1, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1
    b_labels = labels[:bootstrap_size]

    if not cfg.model_cfg.bootstrap_cfg:
        v_b1 = call_model(x_t, t, dt_base_bootstrap, b_labels, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        v_b2 = call_model(x_t2, t2, dt_base_bootstrap, b_labels, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v_target = 0.5 * (v_b1 + v_b2)
    else:
        # num_dt_cfg should also be local; allow zero like the JAX path
        num_dt_cfg = bootstrap_size // max(1, log2_sections)
        x_t_ext = torch.cat([x_t, x_t[:num_dt_cfg]], 0)
        t_ext = torch.cat([t, t[:num_dt_cfg]], 0)
        dt_ext = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], 0)
        labels_ext = torch.cat(
            [b_labels, torch.full((num_dt_cfg,), cfg.runtime_cfg.num_classes, device=device, dtype=torch.long)], 0
        )
        v_raw = call_model(x_t_ext, t_ext, dt_ext, labels_ext, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v_cond, v_uncond = v_raw[:bootstrap_size], v_raw[bootstrap_size:]
        v_cfg = v_uncond + cfg.model_cfg.cfg_scale * (v_cond[:num_dt_cfg] - v_uncond)
        v_b1 = torch.cat([v_cfg, v_cond[num_dt_cfg:]], 0)

        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        x_t2_ext = torch.cat([x_t2, x_t2[:num_dt_cfg]], 0)
        t2_ext = torch.cat([t2, t2[:num_dt_cfg]], 0)
        v2_raw = call_model(x_t2_ext, t2_ext, dt_ext, labels_ext, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v2_cond, v2_uncond = v2_raw[:bootstrap_size], v2_raw[bootstrap_size:]
        v2_cfg = v2_uncond + cfg.model_cfg.cfg_scale * (v2_cond[:num_dt_cfg] - v2_uncond)
        v_b2 = torch.cat([v2_cfg, v2_cond[num_dt_cfg:]], 0)
        v_target = 0.5 * (v_b1 + v_b2)

    v_target = torch.clamp(v_target, -4, 4)
    bst_v, bst_dt, bst_t, bst_xt, bst_l = v_target, dt_base, t, x_t, b_labels

    # ----- flow targets for the *rest of local batch* -----
    rest = B - bootstrap_size
    if rest > 0:
        t_flow = torch.randint(0, cfg.model_cfg.denoise_timesteps, (rest,), generator=gen, device=device).float()
        t_flow = t_flow / float(cfg.model_cfg.denoise_timesteps)
        if force_t != -1:
            t_flow = torch.full_like(t_flow, float(force_t))
        t_flow_full = t_flow.view(rest, 1, 1, 1)

        x1_flow = images[:rest]
        #x0_flow = torch.randn_like(x1_flow, generator=gen)
        x0_flow = torch.randn(x1_flow.shape, dtype=x1_flow.dtype, device=x1_flow.device, generator=gen)
        x_t_flow = (1.0 - (1.0 - 1e-5) * t_flow_full) * x0_flow + t_flow_full * x1_flow
        v_t_flow = x1_flow - (1.0 - 1e-5) * x0_flow

        dt_flow = int(math.log2(cfg.model_cfg.denoise_timesteps))
        dt_base_flow = torch.full((rest,), float(dt_flow), device=device)

        x_t_out = torch.cat([bst_xt, x_t_flow], 0)
        v_t_out = torch.cat([bst_v, v_t_flow], 0)
        t_out = torch.cat([bst_t, t_flow], 0)
        dt_base_out = torch.cat([bst_dt, dt_base_flow], 0)
        labels_out = torch.cat([bst_l, labels[:rest]], 0)
    else:
        # no flow slice on this rank
        x_t_out, v_t_out, t_out, dt_base_out, labels_out = bst_xt, bst_v, bst_t, bst_dt, bst_l

    info['bootstrap_ratio'] = (dt_base_out != int(math.log2(cfg.model_cfg.denoise_timesteps))).float().mean()
    info['v_magnitude_bootstrap'] = bst_v.square().mean().sqrt()
    info['v_magnitude_b1'] = v_b1.square().mean().sqrt()
    info['v_magnitude_b2'] = v_b2.square().mean().sqrt()
    return x_t_out, v_t_out, t_out, dt_base_out, labels_out, info





def get_targets2(cfg, gen, call_model, images, labels, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = cfg.runtime_cfg.batch_size #images.shape[0]

    # Use the *actual* per-step batch size here
    bootstrap_every = int(cfg.model_cfg.bootstrap_every)
    bootstrap_size = max(0, B // max(1, bootstrap_every))  # e.g., B=8, every=4 -> 2
    bootstrap_size = min(bootstrap_size, B)  # never exceed B

    log2_sections = int(math.log2(int(cfg.model_cfg.denoise_timesteps)))

    dt_base = _build_dt_base(
        bootstrap_size=bootstrap_size,
        log2_sections=log2_sections,
        device=device,
        bias=int(cfg.model_cfg.bootstrap_dt_bias)
    )

    if force_dt != -1:
        dt_base = torch.full((bootstrap_size,), float(force_dt), device=device)

    dt = 1.0 / (2.0 ** dt_base)
    dt_base_bootstrap = dt_base + 1.0
    dt_bootstrap = dt / 2.0

    # sample t for bootstrap slice
    # per-sample number of sections (float tensor: e.g., [32., 16., ...])
    dt_sections = (2.0 ** dt_base)  # shape: [bootstrap_size], >= 1

    if bootstrap_size == 0:
        t = dt_base.new_zeros((0,))  # empty, stays consistent
    else:
        # sample u in [0,1) per element
        u = torch.rand(bootstrap_size, device=device, generator=gen)
        # convert to integer bin in [0, dt_sections-1] per element
        t_bins = torch.floor(u * dt_sections).to(torch.int64)
        # extra safety in case of any numeric edge
        t_bins = torch.minimum(t_bins, dt_sections.to(torch.int64) - 1)
        # normalize back to [0,1) grid of that element's section count
        t = t_bins.float() / dt_sections
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(-1, 1, 1, 1)

    # bootstrap pairs
    x1 = images[:bootstrap_size]
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1
    b_labels = labels[:bootstrap_size]

    if not cfg.model_cfg.bootstrap_cfg:
        v_b1 = call_model(x_t, t, dt_base_bootstrap, b_labels, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        v_b2 = call_model(x_t2, t2, dt_base_bootstrap, b_labels, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v_target = 0.5 * (v_b1 + v_b2)
    else:
        num_dt_cfg = max(1, bootstrap_size // log2_sections)
        x_t_ext = torch.cat([x_t, x_t[:num_dt_cfg]], 0)
        t_ext = torch.cat([t, t[:num_dt_cfg]], 0)
        dt_ext = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], 0)
        labels_ext = torch.cat([b_labels, torch.full((num_dt_cfg,), cfg.runtime_cfg.num_classes, device=device, dtype=torch.long)], 0)

        v_raw = call_model(x_t_ext, t_ext, dt_ext, labels_ext, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v_cond, v_uncond = v_raw[:bootstrap_size], v_raw[bootstrap_size:]
        v_cfg = v_uncond + cfg.model_cfg.cfg_scale * (v_cond[:num_dt_cfg] - v_uncond)
        v_b1 = torch.cat([v_cfg, v_cond[num_dt_cfg:]], 0)

        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        x_t2_ext = torch.cat([x_t2, x_t2[:num_dt_cfg]], 0)
        t2_ext = torch.cat([t2, t2[:num_dt_cfg]], 0)
        v2_raw = call_model(x_t2_ext, t2_ext, dt_ext, labels_ext, use_ema=bool(cfg.model_cfg.bootstrap_ema))
        v2_cond, v2_uncond = v2_raw[:bootstrap_size], v2_raw[bootstrap_size:]
        v2_cfg = v2_uncond + cfg.model_cfg.cfg_scale * (v2_cond[:num_dt_cfg] - v2_uncond)
        v_b2 = torch.cat([v2_cfg, v2_cond[num_dt_cfg:]], 0)
        v_target = 0.5 * (v_b1 + v_b2)

    v_target = torch.clamp(v_target, -4, 4)
    bst_v, bst_dt, bst_t, bst_xt, bst_l = v_target, dt_base, t, x_t, b_labels

    # flow targets for the rest
    rest = cfg.runtime_cfg.batch_size - bootstrap_size
    t_flow = torch.randint(0, cfg.model_cfg.denoise_timesteps, (rest,), generator=gen, device=device).float()
    t_flow = t_flow / float(cfg.model_cfg.denoise_timesteps)
    if force_t != -1:
        t_flow = torch.full_like(t_flow, float(force_t))
    t_flow_full = t_flow.view(rest,1,1,1)

    # flow slice
    #x0_flow = torch.randn_like(images[:rest], generator=gen)
    x0_flow = torch.randn((rest, *images.shape[1:]),
                          dtype=images.dtype, device=images.device, generator=gen)
    x1_flow = images[:rest]
    
    print(f"B = {B}")
    print(f"x0_flow = {x0_flow.shape}")
    print(f"x1_flow = {x1_flow.shape}")


    x_t_flow = (1.0 - (1.0 - 1e-5) * t_flow_full) * x0_flow + t_flow_full * x1_flow
    v_t_flow = x1_flow - (1.0 - 1e-5) * x0_flow
    dt_flow = int(math.log2(cfg.model_cfg.denoise_timesteps))
    dt_base_flow = torch.full((rest,), dt_flow, device=device, dtype=torch.float32)

    x_t_out = torch.cat([bst_xt, x_t_flow], 0)
    v_t_out = torch.cat([bst_v, v_t_flow], 0)
    t_out = torch.cat([bst_t, t_flow], 0)
    dt_base_out = torch.cat([bst_dt, dt_base_flow], 0)
    labels_out = torch.cat([bst_l, labels[:rest]], 0)

    info['bootstrap_ratio'] = (dt_base_out != dt_flow).float().mean()
    info['v_magnitude_bootstrap'] = bst_v.square().mean().sqrt()
    info['v_magnitude_b1'] = v_b1.square().mean().sqrt()
    info['v_magnitude_b2'] = v_b2.square().mean().sqrt()
    return x_t_out, v_t_out, t_out, dt_base_out, labels_out, info

import math
import torch

EPS = 1e-5


@torch.no_grad()
def get_targets(
    cfg,
    gen,
    images,
    labels,
    call_model_fn,           # expects: call_model_fn(x, t, dt, labels, train=False)
    step,
    force_t: float = -1.0,
    force_dt: float = -1.0,
):
    """
    Fully continuous shortcut target builder.

    - dt ∈ [dt_min, 1]
    - t ∈ [0, 1 - dt]
    - No discrete levels. We pass/return dt (float) directly.

    Returns:
      x_t_out, v_t_out, t_out, dt_out, labels_out, info
    """
    device = images.device
    B = images.shape[0]

    # Config
    T = int(cfg.model_cfg.denoise_timesteps)                 # only used for defaults/flow grid
    bootstrap_every = cfg.model_cfg.bootstrap_every
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    # dt_min: take from config if provided; else default to 1/T
    dt_min = 1.0 / float(T)

    info = {}

    # ---------------- Split batch ----------------
    B_b = B // bootstrap_every
    B_fm = B - B_b

    # ---------------- Sample continuous dt ----------------
    if force_dt != -1:
        dt_boot = torch.full((B_b,), float(force_dt), device=device)
    else:
        u = torch.rand(B_b, device=device, generator=gen)
        dt_boot = dt_min + (1.0 - dt_min) * u               # Uniform[dt_min, 1]

    # ---------------- Sample continuous t with boundary ----------------
    if force_t != -1:
        t_boot = torch.full((B_b,), float(force_t), device=device)
        # clamp dt so that t + dt <= 1
        dt_boot = torch.minimum(dt_boot, torch.clamp(1.0 - t_boot, min=1e-6))
    else:
        # sample t ∈ [0, 1 - dt]
        tmax = torch.clamp(1.0 - dt_boot, min=0.0)
        u = torch.rand(B_b, device=device, generator=gen)
        t_boot = u * tmax

    # ensure midpoint call (t + dt/2) stays in range (already implied by t<=1-dt)
    half = 0.5 * dt_boot

    # ---------------- Bootstrap (midpoint Heun) ----------------
    x1 = images[:B_b]
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)

    t_full = t_boot.view(B_b, 1, 1, 1)

    x_t = (1.0 - (1.0 - EPS) * t_full) * x0 + t_full * x1
    y_b = labels[:B_b]

    # Teacher calls at dt/2
    v_b1 = call_model_fn(x_t, t_boot, half, y_b, train=False)
    x_t2 = torch.clamp(x_t + half.view(B_b, 1, 1, 1) * v_b1, -4.0, 4.0)
    v_b2 = call_model_fn(x_t2, t_boot + half, half, y_b, train=False)

    v_target = 0.5 * (v_b1 + v_b2)
    v_target = torch.clamp(v_target, -4.0, 4.0)

    # -------- Optional CFG bootstrap (same structure, just pass float dt=half) --------
    if getattr(cfg.model_cfg, "bootstrap_cfg", False):
        # number duplicated (simple heuristic)
        num_dup = max(1, B_b // max(1, int(math.log2(T))))

        x_t_extra   = torch.cat([x_t, x_t[:num_dup]], dim=0)
        t_extra     = torch.cat([t_boot, t_boot[:num_dup]], dim=0)
        half_extra  = torch.cat([half, half[:num_dup]], dim=0)
        y_extra     = torch.cat([y_b, torch.full((num_dup,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, half_extra, y_extra, train=False)
        v_b1_cond, v_b1_uncond = v_b1_raw[:B_b], v_b1_raw[B_b:]
        v_cfg = v_b1_uncond + cfg_scale * (v_b1_cond[:num_dup] - v_b1_uncond)
        v_b1 = torch.cat([v_cfg, v_b1_cond[num_dup:]], dim=0)

        x_t2 = torch.clamp(x_t + half.view(B_b,1,1,1) * v_b1, -4.0, 4.0)
        v_b2_raw = call_model_fn(
            torch.cat([x_t2, x_t2[:num_dup]], dim=0),
            torch.cat([t_boot + half, (t_boot + half)[:num_dup]], dim=0),
            half_extra,
            y_extra,
            train=False
        )
        v_b2_cond, v_b2_uncond = v_b2_raw[:B_b], v_b2_raw[B_b:]
        v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dup] - v_b2_uncond)
        v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dup:]], dim=0)

        v_target = torch.clamp(0.5 * (v_b1 + v_b2), -4.0, 4.0)

    # stash bootstrap outputs
    x_t_boot   = x_t
    v_boot     = v_target
    t_boot_out = t_boot.clone()
    dt_boot_out= dt_boot.clone()
    y_boot     = y_b

    # ---------------- Flow-matching half ----------------
    # label dropout (for CFG training)
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().long()
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    # t on the global grid (or continuous—FM works with either). We'll keep the original grid style:
    t_fm = torch.randint(low=0, high=T, size=(B,), device=device, generator=gen).float() / float(T)
    t_fm_full = t_fm.view(B, 1, 1, 1)

    x0_fm = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen)

    x_t_fm = (1.0 - (1.0 - EPS) * t_fm_full) * x0_fm + t_fm_full * images
    v_t_fm = images - (1.0 - EPS) * x0_fm

    # FM doesn't depend on dt, but the model still needs a dt to encode.
    # Provide a small constant (dt_min) to keep conditioning consistent.
    dt_fm = torch.full((B,), dt_min, device=device)

    # ---------------- Merge ----------------
    x_t_out = torch.cat([x_t_boot,          x_t_fm[:B_fm]], dim=0)
    v_t_out = torch.cat([v_boot,            v_t_fm[:B_fm]], dim=0)
    t_out   = torch.cat([t_boot_out,        t_fm[:B_fm]],   dim=0)
    dt_out  = torch.cat([dt_boot_out,       dt_fm[:B_fm]],  dim=0)   # float dt (no levels)
    labels_out = torch.cat([y_boot,         labels_dropped[:B_fm]], dim=0)

    # telemetry (optional)
    info["dt_min"]   = torch.tensor(dt_min, device=device)
    info["dt_mean"]  = dt_boot_out.mean()
    info["t_mean"]   = t_boot_out.mean()

    return x_t_out, v_t_out, t_out, dt_out, labels_out, info

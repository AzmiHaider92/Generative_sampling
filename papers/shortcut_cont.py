import math
import torch

EPS = 1e-5


def _dt_to_k(dt: torch.Tensor, K_max: int):
    """
    Safe continuous level code: k = -log2(dt) ∈ [0, K_max].
    """
    dt = dt.clamp(min=2.0 ** (-K_max), max=1.0)
    k = -torch.log2(dt)
    return k.clamp(min=0.0, max=float(K_max))


@torch.no_grad()
def get_targets(
    cfg,
    gen,
    images,
    labels,
    call_model_fn,           # expects: call_model_fn(x, t, k, labels, train=False)
    step,
    force_t: float = -1.0,
    force_dt: float = -1.0,
):
    """
    Shortcut target builder with **k-level conditioning**.

    - We still sample/compute dt to advance states, but the model is conditioned on
      k = -log2(dt). The teacher uses k+1 (two half-steps).
    - Flow Matching part uses a constant k corresponding to dt_min.

    Returns:
      x_t_out, v_t_out, t_out, k_out, labels_out, info
    """
    device = images.device
    B = images.shape[0]

    # ---- Config ----
    T = int(cfg.model_cfg.denoise_timesteps)           # e.g., 128
    K_max = max(0, int(round(math.log2(T))))           # e.g., 7 if T=128
    bootstrap_every = cfg.model_cfg.bootstrap_every
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    # dt_min from schedule; keep identical to your original
    dt_min = 1.0 / float(T)

    info = {}

    # ---------------- Split batch ----------------
    B_b = B // bootstrap_every
    B_fm = B - B_b

    # ---------------- Sample dt ----------------
    if force_dt != -1:
        dt_boot = torch.full((B_b,), float(force_dt), device=device)
    else:
        u = torch.rand(B_b, device=device, generator=gen)
        dt_boot = dt_min + (1.0 - dt_min) * u  # Uniform[dt_min, 1]

    # ---------------- Sample t with boundary ----------------
    if force_t != -1:
        t_boot = torch.full((B_b,), float(force_t), device=device)
        # ensure t + dt <= 1
        dt_boot = torch.minimum(dt_boot, torch.clamp(1.0 - t_boot, min=1e-6))
    else:
        tmax = torch.clamp(1.0 - dt_boot, min=0.0)
        u = torch.rand(B_b, device=device, generator=gen)
        t_boot = u * tmax

    # Teacher runs two half-steps (dt/2)
    half = 0.5 * dt_boot

    # ---------------- Build x_t (same as before) ----------------
    x1 = images[:B_b]
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)

    t_full = t_boot.view(B_b, 1, 1, 1)
    x_t = (1.0 - (1.0 - EPS) * t_full) * x0 + t_full * x1
    y_b = labels[:B_b]

    # ---------------- Compute k codes ----------------
    # student level for this dt
    k_stu = _dt_to_k(dt_boot, K_max)           # shape (B_b,)
    # teacher is off-by-one level (dt/2)
    k_teach = (k_stu + 1.0).clamp(max=float(K_max))

    # ---------------- Teacher calls (condition with k, still step by dt) ----------------
    # First half-step at (t, dt/2)
    v_b1 = call_model_fn(x_t, t_boot, k_teach, y_b, train=False)

    # Advance by half *numerically*
    x_t2 = torch.clamp(x_t + half.view(B_b, 1, 1, 1) * v_b1, -4.0, 4.0)

    # Second half-step at (t + dt/2, dt/2)
    v_b2 = call_model_fn(x_t2, t_boot + half, k_teach, y_b, train=False)

    # Midpoint target
    v_target = torch.clamp(0.5 * (v_b1 + v_b2), -4.0, 4.0)

    # -------- Optional CFG bootstrap (same k_teach) --------
    if getattr(cfg.model_cfg, "bootstrap_cfg", False):
        num_dup = max(1, B_b // max(1, int(math.log2(T))))

        x_t_extra  = torch.cat([x_t, x_t[:num_dup]], dim=0)
        t_extra    = torch.cat([t_boot, t_boot[:num_dup]], dim=0)
        k_extra    = torch.cat([k_teach, k_teach[:num_dup]], dim=0)
        y_extra    = torch.cat([y_b,
                                torch.full((num_dup,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, k_extra, y_extra, train=False)
        v_b1_cond, v_b1_uncond = v_b1_raw[:B_b], v_b1_raw[B_b:]
        v_cfg = v_b1_uncond + cfg_scale * (v_b1_cond[:num_dup] - v_b1_uncond)
        v_b1 = torch.cat([v_cfg, v_b1_cond[num_dup:]], dim=0)

        x_t2 = torch.clamp(x_t + half.view(B_b, 1, 1, 1) * v_b1, -4.0, 4.0)

        v_b2_raw = call_model_fn(
            torch.cat([x_t2, x_t2[:num_dup]], dim=0),
            torch.cat([t_boot + half, (t_boot + half)[:num_dup]], dim=0),
            k_extra,
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
    k_boot_out = k_stu.clone()
    y_boot     = y_b

    # ---------------- Flow-matching half (unchanged dynamics) ----------------
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().long()
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    t_fm = torch.randint(low=0, high=T, size=(B,), device=device, generator=gen).float() / float(T)
    t_fm_full = t_fm.view(B, 1, 1, 1)

    x0_fm = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen)
    x_t_fm = (1.0 - (1.0 - EPS) * t_fm_full) * x0_fm + t_fm_full * images
    v_t_fm = images - (1.0 - EPS) * x0_fm

    # Give FM a consistent conditioner: k corresponding to dt_min
    k_fm = torch.full((B,), _dt_to_k(torch.tensor([dt_min], device=device), K_max)[0].item(), device=device)

    # ---------------- Merge ----------------
    x_t_out = torch.cat([x_t_boot,          x_t_fm[:B_fm]], dim=0)
    v_t_out = torch.cat([v_boot,            v_t_fm[:B_fm]], dim=0)
    t_out   = torch.cat([t_boot_out,        t_fm[:B_fm]],   dim=0)
    k_out   = torch.cat([k_boot_out,        k_fm[:B_fm]],   dim=0)   # << return k instead of dt
    labels_out = torch.cat([y_boot,         labels_dropped[:B_fm]], dim=0)

    # telemetry
    info["dt_min"]   = torch.tensor(dt_min, device=device)
    info["dt_mean"]  = dt_boot.mean()
    info["t_mean"]   = t_boot_out.mean()
    info["k_mean"]   = k_out.mean()
    info["K_max"]    = torch.tensor(float(K_max), device=device)

    return x_t_out, v_t_out, t_out, k_out, labels_out, info

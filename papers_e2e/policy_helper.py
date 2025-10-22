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


def sample_t(cfg, gen, images):
    B = images.shape[0]
    device = images.device
    dt_min = 1. / cfg.model_cfg.denoise_timesteps

    x0 = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen) # noise
    u = torch.rand(B, device=device, generator=gen) # rand ~U[0,1]
    t_boot = u * (1-dt_min)
    t_full = t_boot.view(B, 1, 1, 1)
    x_t = (1.0 - (1.0 - EPS) * t_full) * x0 + t_full * images

    return x_t, t_boot


@torch.no_grad()
def cfg_velocity(call_model, x, t, k, labels, *, cfg_scale: float, num_classes: int):
    """
    v(x,t,k,y) with classifier-free guidance, matching your inference code.
      if cfg_scale == 0: v(x,t,k, y_uncond)
      if cfg_scale == 1: v(x,t,k, y_cond)
      else             : v_u + cfg*(v_c - v_u)
    labels: [B] long (true labels); unconditional id is num_classes.
    """
    labels_uncond = torch.full_like(labels, num_classes if num_classes > 1 else 0)
    if cfg_scale == 0.0:
        return call_model(x, t, k, labels_uncond, train=False)

    if cfg_scale == 1.0:
        return call_model(x, t, k, labels, train=False)

    y_u = torch.full_like(labels, num_classes)
    v_u = call_model(x, t, k, y_u, train=False)
    v_c = call_model(x, t, k, labels, train=False)
    return v_u + cfg_scale * (v_c - v_u)


def get_targets(
    cfg,
    labels,
    call_model_fn,         # expects: call_model_fn(x, t, k, labels, train=False)
    x_t,                   # [B, C, H, W]  current state
    t,                     # [B]           current time in [0,1]
    dt,                    # [B]           desired step size (will be clamped to not exceed 1-t)
):
    """
    Bootstrap-only shortcut target builder (no sampling).
    Inputs:
      x_t, t, dt  are provided by the caller.
    Conditioning:
      k = -log2(dt) (continuous); teacher uses k for dt/2 (equivalently k+1, clamped).

    Returns (in this order):
      x_student : [B,C,H,W]  # full step with student velocity at dt
      xh        : [B,C,H,W]  # one half-step (dt/2) using teacher v1
      x2h       : [B,C,H,W]  # two half-steps (midpoint/Heun): xh then another dt/2 with teacher v2
      v_target  : [B,C,H,W]  # midpoint teacher velocity (use for loss)
      k_stu     : [B]        # continuous level code for dt
      labels_out: [B]        # with class dropout applied (for CFG-style training)
      info      : dict       # small telemetry
    """
    device = x_t.device
    B = x_t.shape[0]

    # ---- Config ----
    T       = int(cfg.model_cfg.denoise_timesteps)     # e.g., 128 (only for K_max/telemetry)
    K_max   = max(0, int(round(math.log2(T))))
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    # ---- Safety: ensure step stays within [0,1] horizon ----
    #dt = torch.minimum(dt, torch.clamp(1.0 - t, min=1e-6))  # guarantees t + dt <= 1
    half = 0.5 * dt

    # ---- Compute continuous k codes (student at dt, teacher at dt/2) ----
    k_stu   = _dt_to_k(dt,   K_max)   # student conditioner (full step)
    k_teach = _dt_to_k(half, K_max)   # teacher conditioner (half step)

    # ---- Teacher: first half-step ----
    v_b1 = call_model_fn(x_t, t, k_teach, labels)                    # [B,C,H,W]
    xh   = torch.clamp(x_t + half.view(B,1,1,1) * v_b1, -4.0, 4.0)                # x_{t+dt/2}

    # ---- Teacher: second half-step from midpoint ----
    v_b2 = call_model_fn(xh, t + half, k_teach, labels)              # [B,C,H,W]
    v_target = torch.clamp(0.5 * (v_b1 + v_b2), -4.0, 4.0)                        # midpoint/Heun teacher

    # ---- Student: full step from x_t using student velocity at dt ----
    v_stu = call_model_fn(x_t, t, k_stu, labels)                     # [B,C,H,W]
    x_student = torch.clamp(x_t + dt.view(B,1,1,1) * v_stu, -4.0, 4.0)            # Euler full step

    # ---- Two half-steps state (for reference) ----
    x2h = torch.clamp(xh + half.view(B,1,1,1) * v_b2, -4.0, 4.0)                  # x after two half steps

    # ---- Optional: CFG on teacher (kept identical pattern; uses same k_teach) ----
    if getattr(cfg.model_cfg, "bootstrap_cfg", False):
        # duplicate a small slice as unconditional
        num_dup = max(1, B // max(1, int(math.log2(T))))
        x_t_extra = torch.cat([x_t, x_t[:num_dup]], dim=0)
        t_extra   = torch.cat([t,   t[:num_dup]],   dim=0)
        k_extra   = torch.cat([k_teach, k_teach[:num_dup]], dim=0)
        y_extra   = torch.cat([labels,
                               torch.full((num_dup,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, k_extra, y_extra, train=False)
        v_b1_cond, v_b1_uncond = v_b1_raw[:B], v_b1_raw[B:]
        v_b1 = torch.cat([v_b1_uncond + cfg_scale * (v_b1_cond[:num_dup] - v_b1_uncond),
                          v_b1_cond[num_dup:]], dim=0)
        xh   = torch.clamp(x_t + half.view(B,1,1,1) * v_b1, -4.0, 4.0)

        v_b2_raw = call_model_fn(
            torch.cat([xh, xh[:num_dup]], dim=0),
            torch.cat([t + half, (t + half)[:num_dup]], dim=0),
            k_extra,
            y_extra,
            train=False
        )
        v_b2_cond, v_b2_uncond = v_b2_raw[:B], v_b2_raw[B:]
        v_b2 = torch.cat([v_b2_uncond + cfg_scale * (v_b2_cond[:num_dup] - v_b2_uncond),
                          v_b2_cond[num_dup:]], dim=0)

        v_target = torch.clamp(0.5 * (v_b1 + v_b2), -4.0, 4.0)
        x2h      = torch.clamp(xh + half.view(B,1,1,1) * v_b2, -4.0, 4.0)

    # ---- Label dropout (CFG training trick for the student) ----
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_out = labels.clone().long()
    labels_out[drop] = num_classes if num_classes > 1 else 0

    # ---- Telemetry ----
    info = {
        "dt_mean": dt.mean(),
        "t_mean": t.mean(),
        "k_mean": k_stu.mean(),
        "K_max": torch.tensor(float(K_max), device=device),
    }

    # Return in requested order (x_student, xh, x2h), plus useful training bits
    return x_student, xh, x2h, v_target, k_stu, labels_out, info
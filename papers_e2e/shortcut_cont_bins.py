import math
import torch

EPS = 1e-5


@torch.no_grad()
def get_targets(cfg, gen, images, labels, call_model_fn, step,
                                 force_t: float = -1.0, force_dt_steps: int = -1):
    """
    'shortcut_allbins' target builder.

    t on global grid: t = j / T,  j ∈ {0..T-1}
    dt is any even steps: dt = n / T,  n ∈ {2,4,...,T}, with boundary t + dt <= 1

    Model conditioning uses k_cont = -log2(dt). For the teacher half-step (dt/2),
    just add +1 to k_cont.

    Returns:
      x_t_out, v_t_out, t_out, kcont_out, labels_out, info
        - kcont_out is the continuous level (float), feed to your TimestepEmbedder.
    """
    device = images.device
    B = images.shape[0]
    T = int(cfg.model_cfg.denoise_timesteps)           # e.g., 128
    bootstrap_every = cfg.model_cfg.bootstrap_every
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    info = {}

    # ---------------- 1) Split batch ----------------
    B_b = B // bootstrap_every
    B_fm = B - B_b

    # ---------------- 2) Sample t on GLOBAL grid ----------------
    # We must ensure there is room for at least an even step dt=2/T:
    # so j must be in [0, T-2]
    if force_t != -1:
        # force_t assumed to be a float in [0,1); project to nearest grid cell
        j = torch.clamp((torch.tensor(force_t, device=device) * T).long(), 0, T-2)
        t_boot = j.float() / float(T)
        t_boot = t_boot.expand(B_b)
    else:
        j = torch.randint(low=0, high=T-1, size=(B_b,), generator=gen, device=device)  # 0..T-2 OK; clamp below
        j = torch.clamp(j, max=T-2)
        t_boot = j.float() / float(T)  # [B_b]

    # ---------------- 3) Sample even dt steps with boundary ----------------
    # remaining steps = T - j; even dt_steps ∈ {2,4,..., 2*floor((T - j)/2)}
    rem = T - j                         # [B_b]
    max_even = (rem // 2) * 2          # largest even ≤ remaining
    # guard: if for some reason max_even < 2 (shouldn't happen due to j<=T-2), fix to 2
    max_even = torch.clamp(max_even, min=2)

    if force_dt_steps != -1:
        dt_steps = torch.full_like(max_even, int(force_dt_steps))
        dt_steps = torch.clamp(dt_steps, min=2, max=max_even)
        # make sure it's even
        dt_steps = (dt_steps // 2) * 2
    else:
        # sample k in {1..max_even/2}, then dt_steps = 2*k
        k_even_max = torch.clamp(max_even // 2, min=1)
        # sample uniformly per-example
        # draw big then clamp to make per-example ranges easy
        k_draw = torch.randint(low=1,
                               high=int(k_even_max.max().item()) + 1,
                               size=(B_b,),
                               generator=gen, device=device)
        k_draw = torch.minimum(k_draw, k_even_max)
        dt_steps = 2 * k_draw  # [B_b], even

    dt_boot = dt_steps.float() / float(T)      # [B_b] in (0,1], even fraction
    # full-step boundary guaranteed: t_boot + dt_boot <= 1 by construction

    # continuous level for conditioning
    k_cont = -torch.log2(torch.clamp(dt_boot, min=1e-8))         # [B_b]
    k_cont_half = k_cont + 1.0                                   # because dt/2

    # ---------------- 4) Build bootstrap target (midpoint Heun) ----------------
    x1 = images[:B_b]
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=gen)

    t_boot_full = t_boot.view(B_b, 1, 1, 1)
    x_t = (1.0 - (1.0 - EPS) * t_boot_full) * x0 + t_boot_full * x1
    y_b = labels[:B_b]

    half = 0.5 * dt_boot  # (not used directly by the model; model gets k_cont_half)
    # Teacher at dt/2 (=> k_cont + 1)
    v_b1 = call_model_fn(x_t, t_boot, k_cont_half, y_b, train=False)
    x_t2 = torch.clamp(x_t + half.view(B_b, 1, 1, 1) * v_b1, -4.0, 4.0)
    v_b2 = call_model_fn(x_t2, t_boot + half, k_cont_half, y_b, train=False)

    v_target = 0.5 * (v_b1 + v_b2)
    v_target = torch.clamp(v_target, -4.0, 4.0)

    # ---- Optional CFG (same structure, just pass k_cont_half) ----
    if cfg.model_cfg.bootstrap_cfg:
        # number duplicated (heuristic consistent with original)
        num_dt_cfg = max(1, B_b // max(1, int(math.log2(T))))

        x_t_extra   = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
        t_extra     = torch.cat([t_boot, t_boot[:num_dt_cfg]], dim=0)
        khalf_extra = torch.cat([k_cont_half, k_cont_half[:num_dt_cfg]], dim=0)
        y_extra     = torch.cat([y_b, torch.full((num_dt_cfg,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, khalf_extra, y_extra, train=False)
        v_b1_cond = v_b1_raw[:B_b]
        v_b1_uncond = v_b1_raw[B_b:]
        v_cfg = v_b1_uncond + cfg_scale * (v_b1_cond[:num_dt_cfg] - v_b1_uncond)
        v_b1 = torch.cat([v_cfg, v_b1_cond[num_dt_cfg:]], dim=0)

        x_t2 = torch.clamp(x_t + half.view(B_b,1,1,1) * v_b1, -4.0, 4.0)
        v_b2_raw = call_model_fn(
            torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0),
            torch.cat([t_boot + half, (t_boot + half)[:num_dt_cfg]], dim=0),
            khalf_extra,
            y_extra,
            train=False
        )
        v_b2_cond = v_b2_raw[:B_b]
        v_b2_uncond = v_b2_raw[B_b:]
        v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)

        v_target = torch.clamp(0.5 * (v_b1 + v_b2), -4.0, 4.0)

    # student will be trained at k_cont (the full dt)
    kcont_boot = k_cont.clone()
    t_boot_out = t_boot.clone()
    x_t_boot   = x_t
    v_boot     = v_target
    y_boot     = y_b

    # ---------------- 5) Flow-matching half ----------------
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().long()
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    t_fm = torch.randint(low=0, high=T, size=(B,), device=device, generator=gen).float() / float(T)
    t_fm_full = t_fm.view(B, 1, 1, 1)
    x0_fm = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen)

    x_t_fm = (1.0 - (1.0 - EPS) * t_fm_full) * x0_fm + t_fm_full * images
    v_t_fm = images - (1.0 - EPS) * x0_fm

    # give the model a small even step for conditioning (2/T)
    kcont_fm = torch.full((B,), -math.log2(max(2.0/float(T), 1e-8)), device=device)

    # ---------------- 6) Merge ----------------
    x_t_out   = torch.cat([x_t_boot,        x_t_fm[:B_fm]], dim=0)
    v_t_out   = torch.cat([v_boot,          v_t_fm[:B_fm]], dim=0)
    t_out     = torch.cat([t_boot_out,      t_fm[:B_fm]],   dim=0)
    kcont_out = torch.cat([kcont_boot,      kcont_fm[:B_fm]], dim=0)   # <— use this to embed dt
    labels_out= torch.cat([y_boot,          labels_dropped[:B_fm]], dim=0)

    # (optional) a bit of telemetry
    info['avg_dt_steps_boot'] = dt_steps.float().mean() if 'dt_steps' in locals() else torch.tensor(float('nan'), device=device)
    info['avg_t_boot']        = t_boot_out.mean()
    info['avg_kcont_boot']    = kcont_boot.mean()

    return x_t_out, v_t_out, t_out, kcont_out, labels_out, info

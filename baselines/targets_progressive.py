# targets_progressive_torch.py
import torch
import math

@torch.no_grad()
def get_targets(FLAGS, gen, call_model_teacher, images, labels, step, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = images.shape[0]

    dt_flow = int(math.log2(FLAGS.model['denoise_timesteps']))
    # dt_base decreases as training progresses
    dt_base_scalar = dt_flow - int(step // (FLAGS.max_steps / dt_flow)) - 1
    dt_base = torch.full((B,), dt_base_scalar, device=device, dtype=torch.float32)
    dt = 1.0 / (2.0 ** dt_base)
    dt_base_bootstrap = dt_base + 1.0
    dt_bootstrap = dt / 2.0
    info['dt_base'] = dt_base.mean()

    # sample t on sections
    dt_sections = (2.0 ** dt_base)
    t = torch.randint(0, dt_sections.to(torch.int64), (B,), generator=gen, device=device).float() / dt_sections
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(B,1,1,1)

    x1 = images
    x0 = torch.randn_like(x1, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1

    cfg_scale = FLAGS.model['cfg_scale'] if (dt_base_scalar == dt_flow-1) else 1.0
    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = call_model_teacher(x_t, t, dt_base_bootstrap, labels, use_ema=bool(FLAGS.model['bootstrap_ema']))
        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        v_b2 = call_model_teacher(x_t2, t2, dt_base_bootstrap, labels, use_ema=bool(FLAGS.model['bootstrap_ema']))
        v_target = 0.5 * (v_b1 + v_b2)
    else:
        x_ext = torch.cat([x_t, x_t], 0)
        t_ext = torch.cat([t, t], 0)
        dt_ext = torch.cat([dt_base_bootstrap, dt_base_bootstrap], 0)
        lab_ext = torch.cat([labels, torch.full_like(labels, FLAGS.model['num_classes'])], 0)
        v_raw = call_model_teacher(x_ext, t_ext, dt_ext, lab_ext, use_ema=bool(FLAGS.model['bootstrap_ema']))
        v_cond, v_uncond = v_raw[:B], v_raw[B:]
        v_b1 = v_uncond + cfg_scale * (v_cond - v_uncond)

        t2 = t + dt_bootstrap
        x_t2 = torch.clamp(x_t + dt_bootstrap.view(-1,1,1,1) * v_b1, -4, 4)
        x2_ext = torch.cat([x_t2, x_t2], 0)
        t2_ext = torch.cat([t2, t2], 0)
        v2_raw = call_model_teacher(x2_ext, t2_ext, dt_ext, lab_ext, use_ema=bool(FLAGS.model['bootstrap_ema']))
        v2_cond, v2_uncond = v2_raw[:B], v2_raw[B:]
        v_b2 = v2_uncond + cfg_scale * (v2_cond - v2_uncond)
        v_target = 0.5 * (v_b1 + v_b2)

    info['v_magnitude_bootstrap'] = v_target.square().mean().sqrt()
    info['v_magnitude_b1'] = v_b1.square().mean().sqrt()
    info['v_magnitude_b2'] = v_b2.square().mean().sqrt()
    return x_t, v_target, t, dt_base, labels, info

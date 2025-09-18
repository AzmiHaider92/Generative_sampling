# targets_consistency_distillation_torch.py
import torch
import math

@torch.no_grad()
def get_targets(FLAGS, gen, call_model_teacher, call_model_student_ema, images, labels, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = images.shape[0]

    dt_flow = int(math.log2(FLAGS.model['denoise_timesteps']))
    dt_base = torch.full((B,), dt_flow, device=device, dtype=torch.float32)
    dt_bootstrap = 1.0 / float(FLAGS.model['denoise_timesteps'])

    t = torch.randint(0, FLAGS.model['denoise_timesteps'], (B,), generator=gen, device=device).float()
    t = t / float(FLAGS.model['denoise_timesteps'])
    t_full = t.view(B,1,1,1)

    x1 = images
    x0 = torch.randn_like(x1, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1

    v_b1 = call_model_teacher(x_t, t, dt_base, labels, use_ema=True)
    t2 = t + dt_bootstrap
    x_t2 = torch.clamp(x_t + dt_bootstrap * v_b1, -4, 4)
    v_b2 = call_model_student_ema(x_t2, t2, dt_base, labels, use_ema=True)

    pred_x1 = x_t2 + (1.0 - t2.view(B,1,1,1)) * v_b2
    v_target = (pred_x1 - x_t) / (1.0 - t.view(B,1,1,1))

    info['v_magnitude_bootstrap'] = v_target.square().mean().sqrt()
    info['v_magnitude_b1'] = v_b1.square().mean().sqrt()
    info['v_magnitude_b2'] = v_b2.square().mean().sqrt()
    return x_t, v_target, t, dt_base, labels, info

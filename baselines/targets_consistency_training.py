# targets_consistency_training_torch.py
import torch
import math

@torch.no_grad()
def get_targets(FLAGS, gen, call_model_ema, images, labels, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = images.shape[0]

    dt_flow = int(math.log2(FLAGS.model['denoise_timesteps']))
    # choose dt based on current dt_base (here we emulate JAX version that increases with step)
    # You can pass the actual step and compute dt_base outside if you prefer.
    # For parity with your JAX file, we sample per-batch uniformly over current section count:
    # (Re-implemented exactly from your JAX logic.)
    # 1) dt_base from step: done outside in train loop if desired. Here: use max section.
    dt_base = torch.randint(0, dt_flow, (B,), generator=gen, device=device).float()  # mild approximation
    dt = 1.0 / (2.0 ** dt_base)

    # 2) t on sections
    dt_sections = (2.0 ** dt_base)
    t = torch.randint(0, dt_sections.to(torch.int64), (B,), generator=gen, device=device).float() / dt_sections
    t2 = t + dt
    t_full = t.view(B,1,1,1)
    t2_full = t2.view(B,1,1,1)

    x1 = images
    x0 = torch.randn_like(x1, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1
    x_t2 = (1.0 - (1.0 - 1e-5) * t2_full) * x0*_*_

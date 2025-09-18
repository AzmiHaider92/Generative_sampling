# targets_naive_torch.py
import torch
import math

@torch.no_grad()
def get_targets(FLAGS, gen, call_model, images, labels, force_t=-1, force_dt=-1):
    info = {}
    B = images.shape[0]
    device = images.device

    # label dropout for CFG
    mask = torch.bernoulli(torch.full((B,), FLAGS.model['class_dropout_prob'], device=device)).bool()
    labels_dropped = torch.where(mask, torch.full_like(labels, FLAGS.model['num_classes']), labels)
    info['dropped_ratio'] = (labels_dropped == FLAGS.model['num_classes']).float().mean()

    # t in [0,1]
    t = torch.randint(0, FLAGS.model['denoise_timesteps'], (B,), generator=gen, device=device).float()
    t = t / float(FLAGS.model['denoise_timesteps'])
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(B, 1, 1, 1)

    # flow pairs
    if 'latent' in FLAGS.dataset_name:
        x0 = images[..., :images.shape[-1] // 2]
        x1 = images[..., images.shape[-1] // 2:]
    else:
        x1 = images
        x0 = torch.randn_like(images, generator=gen)
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1
    v_t = x1 - (1.0 - 1e-5) * x0

    dt_flow = int(math.log2(FLAGS.model['denoise_timesteps']))
    dt_base = torch.full((B,), dt_flow, device=device, dtype=torch.float32)

    return x_t, v_t, t, dt_base, labels_dropped, info

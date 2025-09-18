# targets_livereflow_torch.py
import torch
import math

@torch.no_grad()
def get_targets(FLAGS, gen, call_model, images, labels, force_t=-1, force_dt=-1):
    device = images.device
    info = {}
    B = images.shape[0]

    # Flow batch
    mask = torch.bernoulli(torch.full((B,), FLAGS.model['class_dropout_prob'], device=device)).bool()
    labels_dropped = torch.where(mask, torch.full_like(labels, FLAGS.model['num_classes']), labels)
    info['dropped_ratio'] = (labels_dropped == FLAGS.model['num_classes']).float().mean()

    t = torch.randint(0, FLAGS.model['denoise_timesteps'], (B,), generator=gen, device=device).float()
    t = t / float(FLAGS.model['denoise_timesteps'])
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(B,1,1,1)

    x0 = torch.randn_like(images, generator=gen)
    x1 = images
    x_t = (1.0 - (1.0 - 1e-5) * t_full) * x0 + t_full * x1
    v_t = x1 - (1.0 - 1e-5) * x0
    dt_flow = int(math.log2(FLAGS.model['denoise_timesteps']))
    dt_base = torch.full((B,), dt_flow, device=device, dtype=torch.float32)

    # Reflow bootstrap slice
    bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    x0_rf = torch.randn_like(images, generator=gen)[:bootstrap_size]
    x = x0_rf
    t_iter = torch.zeros(bootstrap_size, device=device, dtype=torch.float32)
    labels_uncond = torch.full((bootstrap_size,), FLAGS.model['num_classes'], device=device, dtype=torch.long)

    for _ in range(8):
        if FLAGS.model['cfg_scale'] == 0:
            v = call_model(x, t_iter, dt_base[:bootstrap_size], labels_uncond, use_ema=False)
        else:
            x_ext = torch.cat([x, x], 0)
            t_ext = torch.cat([t_iter, t_iter], 0)
            dt_ext = torch.cat([dt_base[:bootstrap_size], dt_base[:bootstrap_size]], 0)
            labels_ext = torch.cat([labels[:bootstrap_size], labels_uncond], 0)
            v_all = call_model(x_ext, t_ext, dt_ext, labels_ext, use_ema=False)
            v_cond, v_uncond = v_all[:bootstrap_size], v_all[bootstrap_size:]
            v = v_uncond + FLAGS.model['cfg_scale'] * (v_cond - v_uncond)
        t_iter = t_iter + 1.0 / 8.0
        x = x + (1.0 / 8.0) * v

    v_reflow = (x - x0_rf)
    dt_base_reflow = torch.zeros(bootstrap_size, device=device, dtype=torch.float32)
    t_reflow = torch.randint(0, FLAGS.model['denoise_timesteps'], (bootstrap_size,), generator=gen, device=device).float()
    t_reflow = t_reflow / float(FLAGS.model['denoise_timesteps'])
    t_reflow_full = t_reflow.view(bootstrap_size,1,1,1)
    x_t_reflow = (1.0 - (1.0 - 1e-5) * t_reflow_full) * x0_rf + t_reflow_full * x

    # Combine
    x_t = torch.cat([x_t_reflow, x_t[:-bootstrap_size]], 0)
    v_t = torch.cat([v_reflow, v_t[:-bootstrap_size]], 0)
    t = torch.cat([t_reflow, t[:-bootstrap_size]], 0)
    dt_base = torch.cat([dt_base_reflow, dt_base[:-bootstrap_size]], 0)
    labels_dropped = torch.cat([labels[:bootstrap_size], labels_dropped[:-bootstrap_size]], 0)

    return x_t, v_t, t, dt_base, labels_dropped, info

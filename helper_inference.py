# helper_inference_torch.py
import os
import numpy as np
import torch
import tqdm

@torch.no_grad()
def do_inference(
    FLAGS,
    model,                    # torch.nn.Module (maybe DDP-wrapped)
    ema_model,                # torch.nn.Module or None
    step,                     # int or None
    dataset_iter,
    dataset_valid_iter,
    vae_encode=None,
    vae_decode=None,
    get_fid_activations=None,
    imagenet_labels=None,
    visualize_labels=None,
    fid_from_stats=None,
    truth_fid_stats=None,
):
    device = next(model.parameters()).device
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Pull one batch for shape; JAX also takes shapes from current dataset. :contentReference[oaicite:10]{index=10}
    batch_images, batch_labels = next(dataset_iter)
    valid_images, valid_labels = next(dataset_valid_iter)
    if FLAGS.model.use_stable_vae and vae_encode is not None:
        batch_images = vae_encode(batch_images)
        valid_images = vae_encode(valid_images)

    images_shape = batch_images.shape
    denoise_timesteps = getattr(FLAGS, "inference_timesteps", 128)
    num_generations = getattr(FLAGS, "inference_generations", 4096)
    cfg_scale = getattr(FLAGS, "inference_cfg_scale", 1.0)

    print(f"Sampling cfg={cfg_scale} with T={denoise_timesteps} for {num_generations} images")  # :contentReference[oaicite:11]{index=11}

    B = FLAGS.batch_size
    delta_t = 1.0 / denoise_timesteps
    all_x0, all_x1, all_labels = [], [], []

    # Internal callable to run model (EMA if available) like your JAX call_model() wrapper. :contentReference[oaicite:12]{index=12}
    def call_model(x, t_vector, dt_base, labels, use_ema=True):
        m = ema_model if (use_ema and getattr(FLAGS.model, "use_ema", 0)) else model
        # model forward expects BHWC and returns v_pred (BHWC)
        v_pred, _, _ = m(x, t_vector, dt_base, labels, train=False, return_activations=True)
        return v_pred

    # Choose dt_base per JAX logic: smallest dt for naive; otherwise from inference T. :contentReference[oaicite:13]{index=13}
    def make_dt_base(n):
        if FLAGS.model.train_type == 'naive':
            dt_flow = int(np.log2(FLAGS.model['denoise_timesteps']))
        else:
            dt_flow = int(np.log2(denoise_timesteps))
        return torch.full((n,), dt_flow, device=device, dtype=torch.float32)

    for fid_it in tqdm.tqdm(range(num_generations // B)):
        # New noise + labels every chunk, like JAX: x ~ N(0,I), labels ~ Uniform classes. :contentReference[oaicite:14]{index=14}
        x = torch.randn(images_shape, device=device)
        labels = torch.randint(0, FLAGS.model['num_classes'], (images_shape[0],), device=device, dtype=torch.long)
        all_x0.append(x.detach().cpu().numpy())

        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps
            t_vector = torch.full((images_shape[0],), t, device=device, dtype=torch.float32)
            dt_base = make_dt_base(images_shape[0])

            if cfg_scale == 1:
                v = call_model(x, t_vector, dt_base, labels, use_ema=True)
            elif cfg_scale == 0:
                labels_uncond = torch.full_like(labels, FLAGS.model['num_classes'])
                v = call_model(x, t_vector, dt_base, labels_uncond, use_ema=True)
            else:
                labels_uncond = torch.full_like(labels, FLAGS.model['num_classes'])
                v_uncond = call_model(x, t_vector, dt_base, labels_uncond, use_ema=True)
                v_cond = call_model(x, t_vector, dt_base, labels, use_ema=True)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)  # CFG mix :contentReference[oaicite:15]{index=15}

            if FLAGS.model.train_type == 'consistency':
                # Consistency step: x1pred = x + v*(1-t); then blend with fresh eps. :contentReference[oaicite:16]{index=16}
                eps = torch.randn_like(x)
                x1pred = x + v * (1.0 - t)
                x = x1pred * (t + delta_t) + eps * (1.0 - t - delta_t)
            else:
                # Euler update
                x = x + v * delta_t  # :contentReference[oaicite:17]{index=17}

        all_x1.append(x.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        # optional on-the-fly rendering for small N: decode and stash
        # (Your JAX version conditionally saved render arrays) :contentReference[oaicite:18]{index=18}

    # Optionally decode to image space
    # NOTE: generation x is in latent/image space depending on training; if VAE used, decode to [-1,1]
    if FLAGS.model.use_stable_vae and vae_decode is not None:
        # Decode last chunk just to verify pipeline
        x_vis = vae_decode(torch.tensor(all_x1[-1], device=device))
        x_vis = (x_vis * 0.5 + 0.5).clamp(0, 1)

    # Optional FID computation against provided stats (approximate, subset)
    if get_fid_activations and truth_fid_stats is not None:
        # Collect a reasonable subset to keep memory sane
        subset = min(8192, len(all_x1) * B)
        xs = torch.tensor(np.concatenate(all_x1, axis=0)[:subset], device=device)
        if FLAGS.model.use_stable_vae and vae_decode is not None:
            xs = vae_decode(xs)
        xs = (xs * 0.5 + 0.5).clamp(0, 1)
        acts = []
        for i in range(0, xs.shape[0], 64):
            acts.append(get_fid_activations(xs[i:i+64]).cpu())
        acts = torch.cat(acts, dim=0).numpy()
        mu, sigma = acts.mean(0), np.cov(acts, rowvar=False)
        fid = fid_from_stats(mu, sigma, truth_fid_stats['mu'], truth_fid_stats['sigma'])
        print(f"[inference] FID (subset): {fid:.2f}")

    # Optionally save raw arrays for later analysis
    if getattr(FLAGS, "save_dir", None):
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        np.save(os.path.join(FLAGS.save_dir, "x0.npy"), np.concatenate(all_x0, axis=0))
        np.save(os.path.join(FLAGS.save_dir, "x1.npy"), np.concatenate(all_x1, axis=0))
        np.save(os.path.join(FLAGS.save_dir, "labels.npy"), np.concatenate(all_labels, axis=0))

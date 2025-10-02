from torchvision.utils import save_image
import torchvision.utils as vutils
import wandb
import numpy as np
import torch
import tqdm
from torchmetrics.image import FrechetInceptionDistance
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def do_inference(
    cfg,
    model,                    # torch.nn.Module (maybe DDP-wrapped)
    ema_model,                # torch.nn.Module or None
    dataset_iter,
    vae_encode=None,
    vae_decode=None,
    num_generations=8,
    calc_fid=False,
    step=0
):
    device = next(model.parameters()).device
    was_training = model.training
    was_training_ema = (ema_model.training if ema_model is not None else None)
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # for fid calc
    fid = None
    if (cfg.runtime_cfg.fid_stats is not None) and calc_fid:
        # Metric on GPU; disable AMP for numerical stability
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,  # inputs must be [0,1]
            input_img_size=(3, 299, 299),
            antialias=True,
        ).to(device)

        # Load REAL stats (only the real_* buffers)
        d = torch.load(cfg.runtime_cfg.fid_stats, map_location=device)
        with torch.no_grad():
            fid.real_features_sum.copy_(d["real_features_sum"].to(device))
            fid.real_features_cov_sum.copy_(d["real_features_cov_sum"].to(device))
            fid.real_features_num_samples.copy_(d["real_features_num_samples"].to(device))

    # Pull one batch for shape; JAX also takes shapes from current dataset. :contentReference[oaicite:10]{index=10}
    batch_images, batch_labels = next(dataset_iter)
    if cfg.model_cfg.use_stable_vae and vae_encode is not None:
        batch_images = vae_encode(batch_images)

    images_shape = batch_images.shape
    denoise_timesteps = cfg.runtime_cfg.inference_timesteps
    #num_generations = cfg.runtime_cfg.inference_generations
    cfg_scale = cfg.runtime_cfg.inference_cfg_scale
    nimgs_2_vis, imgs_2_vis = 0, []

    print(f"Sampling cfg={cfg_scale} with T={denoise_timesteps} for {num_generations} images")  # :contentReference[oaicite:11]{index=11}

    B = images_shape[0]
    delta_t = 1.0 / denoise_timesteps
    #all_x0, all_x1, all_labels = [], [], []

    # Internal callable to run model (EMA if available) like your JAX call_model() wrapper. :contentReference[oaicite:12]{index=12}
    def call_model(x, t_vector, dt_base, labels, use_ema=True):
        m = ema_model if (use_ema and getattr(cfg.model_cfg, "use_ema", 0)) else model
        # model forward expects BHWC and returns v_pred (BHWC)
        v_pred = m(x, t_vector, dt_base, labels, train=False, return_activations=True)
        return v_pred

    # Choose dt_base per JAX logic: smallest dt for naive; otherwise from inference T. :contentReference[oaicite:13]{index=13}
    def make_dt_base(n):
        if cfg.model_cfg.train_type == 'naive':
            dt_flow = int(np.log2(cfg.model_cfg.denoise_timesteps))
        else:
            dt_flow = int(np.log2(denoise_timesteps))
        return torch.full((n,), dt_flow, device=device, dtype=torch.float32)

    for fid_it in tqdm.tqdm(range(max(num_generations // B, 1))):
        # New noise + labels every chunk, like JAX: x ~ N(0,I), labels ~ Uniform classes. :contentReference[oaicite:14]{index=14}
        x = torch.randn(images_shape, device=device)
        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long)
        #all_x0.append(x.detach().cpu().numpy())

        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps
            t_vector = torch.full((B,), t, device=device, dtype=torch.float32)
            dt_base = make_dt_base(B)

            if cfg_scale == 1:
                v = call_model(x, t_vector, dt_base, labels, use_ema=True)
            elif cfg_scale == 0:
                labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes)
                v = call_model(x, t_vector, dt_base, labels_uncond, use_ema=True)
            else:
                labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes)
                v_uncond = call_model(x, t_vector, dt_base, labels_uncond, use_ema=True)
                v_cond = call_model(x, t_vector, dt_base, labels, use_ema=True)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)  # CFG mix :contentReference[oaicite:15]{index=15}

            if cfg.model_cfg.train_type == 'consistency':
                # Consistency step: x1pred = x + v*(1-t); then blend with fresh eps. :contentReference[oaicite:16]{index=16}
                eps = torch.randn_like(x)
                x1pred = x + v * (1.0 - t)
                x = x1pred * (t + delta_t) + eps * (1.0 - t - delta_t)
            else:
                # Euler update
                x = x + v * delta_t

        x1 = x.detach().clone()
        if cfg.model_cfg.use_stable_vae and vae_decode is not None:
            # Decode last chunk just to verify pipeline
            x1 = x1.to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae_decode(x1)
            x_vis = (x_vis + 1.0) * 0.5
            x1 = x_vis.clamp(0.0, 1.0)
            #x1 = x_vis.permute(0, 3, 1, 2)

        #all_x1.append(x1.detach().cpu().numpy())
        #all_labels.append(labels.detach().cpu().numpy())

        if nimgs_2_vis < 8: # visualize just 8, hardcoded
            imgs_2_vis.append(x1)
            nimgs_2_vis += x1.shape[0]

        if fid is not None:
            # update FAKE side
            fid.update(x1, real=False)

    # Optional FID computation against provided stats (approximate, subset)
    if fid is not None:
        # Compute FID
        with torch.cuda.amp.autocast(enabled=False):
            score = fid.compute().item()
        print(f"============== FID = {score:.4f}  (N={num_generations}) ====================")

    imgs = torch.cat(imgs_2_vis, dim=0)
    imgs = imgs[:8]

    grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
    save_image(grid, os.path.join(cfg.runtime_cfg.save_dir, f"generated_img_step{step}.png"))
    wandb.log({"Generated samples": wandb.Image(grid)})
    # Optionally save raw arrays for later analysis
    #if cfg.runtime_cfg.save_dir is not None:
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "x0.npy"), np.concatenate(all_x0, axis=0))
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "x1.npy"), np.concatenate(all_x1, axis=0))
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "labels.npy"), np.concatenate(all_labels, axis=0))

    # restore original modes
    if was_training:
        model.train()
    else:
        model.eval()
    if ema_model is not None:
        if was_training_ema:
            ema_model.train()
        else:
            ema_model.eval()


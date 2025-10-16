import math
import torch
import torch.distributed as dist
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
import wandb
import tqdm
from torchmetrics.image import FrechetInceptionDistance
import os
from utils.datasets import get_dataset as get_dataset_iter
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

fid_image_size = 256


# =====================
# VALIDATE (TRM-compatible)
# =====================
import os, math, tqdm, torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance

@torch.no_grad()
def validate(
    cfg,
    ema_model,                # torch.nn.Module (TRM or plain)
    dataset_iter,
    vae=None,
    num_generations=8,
    calc_fid=False,
    step=0,
):
    """TRM-compatible validate: uses v_ref if model returns dict; falls back to tensor.
    Supports CFG at inference. Handles VAE decode and FID.
    """
    # Pull a batch just for shapes (and labels sampling range)
    batch_images, batch_labels = next(dataset_iter)
    device = batch_images.device

    if getattr(cfg.model_cfg, 'use_stable_vae', False) and (vae is not None):
        # only for shape estimation (latents)
        batch_images = vae.encode(batch_images)
    images_shape = batch_images.shape  # (B,C,H,W)
    B = images_shape[0]

    # ----- Inference knobs -----
    denoise_timesteps = int(cfg.runtime_cfg.inference_timesteps)
    cfg_scale = float(cfg.runtime_cfg.inference_cfg_scale)
    trm_temp  = float(getattr(cfg.runtime_cfg, 'trm_temperature_inference',
                              getattr(cfg.model_cfg, 'trm_temperature', 5.0)))
    trm_top1  = bool(getattr(cfg.runtime_cfg, 'use_trm_top1_inference', False))

    dt = 1.0 / denoise_timesteps
    K = int(math.log2(denoise_timesteps))
    k_code = torch.full((B,), float(K), device=device, dtype=torch.float32)  # sentinel level

    print(f"Sampling cfg={cfg_scale} with T={denoise_timesteps} for {num_generations} images")

    was_training = ema_model.training
    ema_model.eval()

    def _call_model(x, t_vec, k, labels):
        # Try TRM signature (temperature/top1); fall back to plain
        try:
            out = ema_model(x, t_vec, k, labels, train=False,
                            temperature=trm_temp, top1=trm_top1)
        except TypeError:
            out = ema_model(x, t_vec, k, labels, train=False)
        # If model returns dict (TRM), use v_ref; else assume tensor velocity
        if isinstance(out, dict):
            return out['v_ref']
        return out

    # ----- Optional FID -----
    fid = None
    if (getattr(cfg.runtime_cfg, 'fid_stats', None) is not None) and calc_fid:
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,                # inputs must be in [0,1]
            input_img_size=(3, fid_image_size, fid_image_size),
            antialias=True,
        ).to(device)
        # Load REAL stats
        d = torch.load(cfg.runtime_cfg.fid_stats, map_location=device)
        fid.real_features_sum.copy_(d['real_features_sum'].to(device))
        fid.real_features_cov_sum.copy_(d['real_features_cov_sum'].to(device))
        fid.real_features_num_samples.copy_(d['real_features_num_samples'].to(device))

    # ----- Sampling loop -----
    n_per_iter = max(num_generations // B, 1)
    imgs_2_vis = []
    nimgs_2_vis = 0

    for _ in tqdm.tqdm(range(n_per_iter)):
        # Random class labels for conditional sampling (and uncond token)
        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0)

        # Start from Gaussian noise (same space as training)
        x = torch.randn(images_shape, device=device)

        for ti in range(denoise_timesteps):
            t = (ti + 0.5) / denoise_timesteps
            t_vec = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 1:
                v = _call_model(x, t_vec, k_code, labels)
            elif cfg_scale == 0:
                v = _call_model(x, t_vec, k_code, labels_uncond)
            else:
                v_u = _call_model(x, t_vec, k_code, labels_uncond)
                v_c = _call_model(x, t_vec, k_code, labels)
                v = v_u + cfg_scale * (v_c - v_u)

            # Euler update (Heun optional – add PC if you want)
            x = x + v * dt

        # Convert to [0,1] images for viz/FID
        x1 = x.detach()
        if getattr(cfg.model_cfg, 'use_stable_vae', False) and (vae is not None):
            # Decode latents → pixel space in [-1,1], then map to [0,1]
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)
            x01 = (x_vis.clamp(-1.0, 1.0) + 1.0) * 0.5
        else:
            # If training in pixel space, ensure proper range
            x01 = (x1.clamp(-1.0, 1.0) + 1.0) * 0.5

        # Collect a few for visualization
        if nimgs_2_vis < 8:
            imgs_2_vis.append(x01)
            nimgs_2_vis += x01.shape[0]

        if fid is not None:
            fid.update(x01, real=False)

    # ----- FID compute (subset) -----
    if fid is not None:
        score = fid.compute().item()
        print(f"============== FID = {score:.4f}  (N={num_generations}) ====================")

    # ----- Visualization grid -----
    if len(imgs_2_vis) > 0:
        imgs = torch.cat(imgs_2_vis, dim=0)[:8]
        grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
        out_path = os.path.join(cfg.runtime_cfg.save_dir,
                                f"generated_img_step{step}_cfg{cfg_scale}_denoise{denoise_timesteps}.png")
        save_image(grid, out_path)
        try:
            import wandb
            wandb.log({"Generated samples": wandb.Image(grid)})
        except Exception:
            pass

    # restore mode
    ema_model.train() if was_training else ema_model.eval()


@torch.no_grad()
def inference(
    cfg,
    ema_model,                   # model on the correct device
    vae=None,                    # optional StableVAE wrapper
    num_generations=50_000,      # TOTAL images across all GPUs
    fid_stats_path=None,         # path to precomputed REAL stats (required for FID)
):
    """
    Multi-GPU FID inference compatible with TRM outputs.
    - Uses refined velocity (v_ref) if model returns a dict, else raw tensor.
    - Supports TRM inference knobs: temperature & top1 gating via cfg.runtime_cfg.
    - Expects sampling domain to match training domain (pixel or VAE latent),
      then converts to [0,1] in pixel space for FID.
    """
    # -------- DDP env --------
    is_dist = dist.is_available() and dist.is_initialized()
    world   = dist.get_world_size() if is_dist else 1
    rank    = dist.get_rank() if is_dist else 0

    if is_dist:
        dist.barrier()

    # -------- Build a tiny iterator for shapes --------
    per_rank_bs = max(1, cfg.runtime_cfg.batch_size // world) if is_dist else cfg.runtime_cfg.batch_size
    dataset_iter = get_dataset_iter(
        cfg.runtime_cfg.dataset_name,
        cfg.runtime_cfg.dataset_root_dir,
        per_rank_bs, True, cfg.runtime_cfg.debug_overfit
    )
    batch_images, batch_labels = next(dataset_iter)
    device = batch_images.device

    if getattr(cfg.model_cfg, 'use_stable_vae', False) and (vae is not None):
        batch_images = vae.encode(batch_images)   # latents for shape
    images_shape = batch_images.shape            # (B,C,H,W)
    B = images_shape[0]

    # -------- Quotas --------
    n_total = int(num_generations)
    n_local = (n_total + world - 1) // world if is_dist else n_total
    iters   = (n_local + B - 1) // B
    if rank == 0:
        print(f"[inference] world={world} | total={n_total} | per-rank={n_local} | "
              f"T={cfg.runtime_cfg.inference_timesteps} | cfg={cfg.runtime_cfg.inference_cfg_scale} | B={B}")

    # -------- Model mode --------
    was_training = ema_model.training
    ema_model.eval()

    # -------- FID setup --------
    fid_score = None
    fid = None
    if fid_stats_path is not None:
        try:
            fid = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
                input_img_size=(3, fid_image_size, fid_image_size),
                antialias=True,
                sync_on_compute=False,
            ).to(device)
        except TypeError:
            fid = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
                input_img_size=(3, fid_image_size, fid_image_size),
                antialias=True,
            ).to(device)
            if hasattr(fid, 'sync_on_compute'):
                fid.sync_on_compute = False

        d = torch.load(fid_stats_path, map_location=device)
        fid.real_features_sum.copy_(d['real_features_sum'].to(device))
        fid.real_features_cov_sum.copy_(d['real_features_cov_sum'].to(device))
        fid.real_features_num_samples.copy_(d['real_features_num_samples'].to(device))

    # -------- TRM inference knobs --------
    trm_temp  = float(getattr(cfg.runtime_cfg, 'trm_temperature_inference',
                              getattr(cfg.model_cfg, 'trm_temperature', 5.0)))
    trm_top1  = bool(getattr(cfg.runtime_cfg, 'use_trm_top1_inference', False))

    # -------- Helpers --------
    def _call_model(x_bchw, t_vec, k, labels):
        # Try TRM signature; fall back to plain
        try:
            out = ema_model(x_bchw, t_vec, k, labels, train=False,
                            temperature=trm_temp, top1=trm_top1)
        except TypeError:
            out = ema_model(x_bchw, t_vec, k, labels, train=False)
        if isinstance(out, dict):
            return out['v_ref']
        return out

    denoise_T = int(cfg.runtime_cfg.inference_timesteps)
    K = int(math.log2(denoise_T))
    dt = 1.0 / denoise_T
    cfg_scale = float(cfg.runtime_cfg.inference_cfg_scale)

    k_code = torch.full((B,), float(K), device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(1234 + rank)

    preview = []
    generated = 0
    pbar = tqdm.tqdm(range(iters), disable=(rank != 0))

    for _ in pbar:
        take = min(B, n_local - generated)
        if take <= 0:
            break

        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long, generator=gen)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0)
        x = torch.randn(images_shape, device=device, generator=gen)

        # Euler sampling (use t=(ti+0.5)/T to mirror training sampling)
        for ti in range(denoise_T):
            t = (ti + 0.5) / denoise_T
            t_vec = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 0:
                v = _call_model(x, t_vec, k_code, labels_uncond)
            elif cfg_scale == 1:
                v = _call_model(x, t_vec, k_code, labels)
            else:
                v_u = _call_model(x, t_vec, k_code, labels_uncond)
                v_c = _call_model(x, t_vec, k_code, labels)
                v = v_u + cfg_scale * (v_c - v_u)

            x = x + v * dt

        # Map to [0,1] BCHW for FID
        x1 = x[:take].detach()
        if getattr(cfg.model_cfg, 'use_stable_vae', False) and (vae is not None):
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)  # might be BCHW or BHWC
            if x_vis.dim() == 4 and x_vis.shape[1] in (1, 3, 4):
                x_bchw = x_vis
            elif x_vis.dim() == 4 and x_vis.shape[-1] in (1, 3, 4):
                x_bchw = x_vis.permute(0, 3, 1, 2).contiguous()
            else:
                x_bchw = x_vis  # best-effort
        else:
            x_bchw = x1
        x01 = (x_bchw.clamp(-1.0, 1.0) + 1.0) * 0.5

        if fid is not None:
            fid.update(x01, real=False)

        if rank == 0 and len(preview) < 8:
            need = 8 - len(preview)
            preview.extend([img for img in x01[:need]])

        generated += take
        if rank == 0:
            pbar.set_postfix_str(f"gen={generated}/{n_local}")

    # -------- Reduce FID fake stats across ranks --------
    if fid is not None:
        if is_dist:
            for name in [
                'fake_features_sum', 'fake_features_cov_sum', 'fake_features_num_samples'
            ]:
                if hasattr(fid, name):
                    dist.all_reduce(getattr(fid, name), op=dist.ReduceOp.SUM)
        if (not is_dist) or rank == 0:
            torch.cuda.synchronize()
            with torch.amp.autocast('cuda', enabled=False):
                fid_score = fid.compute().item()
                print(f"FID: {fid_score:.4f}")
            torch.cuda.synchronize()
            try:
                import wandb
                if hasattr(wandb, 'log') and wandb.run is not None:
                    wandb.log({"metrics/FID": fid_score})
            except Exception:
                pass

    # -------- Save a small grid (rank 0) --------
    if rank == 0 and len(preview) > 0:
        imgs = torch.stack(preview, dim=0)
        grid = make_grid(imgs, nrow=4, padding=2, normalize=False)
        os.makedirs(cfg.runtime_cfg.save_dir, exist_ok=True)
        tag = f"_FID{fid_score:.4f}" if fid_score is not None else ""
        out_path = os.path.join(
            cfg.runtime_cfg.save_dir,
            f"generated_step{getattr(cfg.runtime_cfg,'global_step',0)}_cfg{cfg.runtime_cfg.inference_cfg_scale}_T{cfg.runtime_cfg.inference_timesteps}{tag}.png"
        )
        save_image(grid, out_path)
        try:
            import wandb
            cap = f"FID: {fid_score:.4f}" if fid_score is not None else f"Samples cfg={cfg.runtime_cfg.inference_cfg_scale} T={cfg.runtime_cfg.inference_timesteps}"
            if hasattr(wandb, 'log') and wandb.run is not None:
                wandb.log({"Generated samples": wandb.Image(grid, caption=cap)})
        except Exception:
            pass

    # restore mode
    ema_model.train() if was_training else ema_model.eval()
    return fid_score

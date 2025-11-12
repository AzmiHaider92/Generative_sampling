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


@torch.no_grad()
def validate(
    cfg,
    ema_model,                # torch.nn.Module or None
    dataset_iter,
    vae=None,
    num_generations=8,
    calc_fid=False,
    step=0,
):
    # Pull one batch for shape; JAX also takes shapes from current dataset. :contentReference[oaicite:10]{index=10}
    batch_images, batch_labels = next(dataset_iter)
    if cfg.model_cfg.use_stable_vae:
        batch_images = vae.encode(batch_images)
    images_shape = batch_images.shape

    device = batch_images.device
    B = images_shape[0]

    denoise_timesteps = cfg.runtime_cfg.inference_timesteps
    cfg_scale = cfg.runtime_cfg.inference_cfg_scale
    dt = 1.0 / denoise_timesteps
    K = int(math.log2(denoise_timesteps))  # max level
    if cfg.model_cfg.train_type == "meanflows":
        K = 1
        denoise_timesteps = 1
        dt = 1.0
    k = torch.full((B,), float(K), device=device, dtype=torch.float32)  # per-sample level code (sentinel)

    print(
        f"Sampling cfg={cfg_scale} with T={denoise_timesteps} for {num_generations} images")

    was_training = ema_model.training
    ema_model.eval()

    def call_model(x, t_vector, k, labels):
        m = ema_model
        # model forward expects BHWC and returns v_pred (BHWC)
        v_pred = m(x, t_vector, k, labels, train=False)
        return v_pred

    # for fid calc
    fid = None
    if (cfg.runtime_cfg.fid_stats is not None) and calc_fid:
        # Metric on GPU; disable AMP for numerical stability
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,  # inputs must be [0,1]
            input_img_size=(3, fid_image_size, fid_image_size),
            antialias=True,
        ).to(device)

        # Load REAL stats (only the real_* buffers)
        d = torch.load(cfg.runtime_cfg.fid_stats, map_location=device)
        with torch.no_grad():
            fid.real_features_sum.copy_(d["real_features_sum"].to(device))
            fid.real_features_cov_sum.copy_(d["real_features_cov_sum"].to(device))
            fid.real_features_num_samples.copy_(d["real_features_num_samples"].to(device))

    nimgs_2_vis, imgs_2_vis = 0, []
    for fid_it in tqdm.tqdm(range(max(num_generations // B, 1))):
        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0)
        x = torch.randn(images_shape, device=device)

        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps
            t_vector = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 1:
                v = call_model(x, t_vector, k, labels)
            elif cfg_scale == 0:
                v = call_model(x, t_vector, k, labels_uncond)
            else:
                v_uncond = call_model(x, t_vector, k, labels_uncond)
                v_cond = call_model(x, t_vector, k, labels)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)  # CFG mix :contentReference[oaicite:15]{index=15}

            #if cfg.model_cfg.train_type == 'consistency':
            #    # Consistency step: x1pred = x + v*(1-t); then blend with fresh eps. :contentReference[oaicite:16]{index=16}
            #    eps = torch.randn_like(x)
            #    x1pred = x + v * (1.0 - t)
            #    x = x1pred * (t + dt) + eps * (1.0 - t - dt)
            #else:
                # Euler update
            x = x + v * dt

        x1 = x.detach().clone()
        if cfg.model_cfg.use_stable_vae:
            # Decode last chunk just to verify pipeline
            x1 = x1.to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)
            x_vis = x_vis.clamp(-1.0, 1.0)
            x1 = (x_vis + 1.0) * 0.5
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
        with torch.amp.autocast('cuda', enabled=False):
            score = fid.compute().item()
        print(f"============== FID = {score:.4f}  (N={num_generations}) ====================")

    imgs = torch.cat(imgs_2_vis, dim=0)
    imgs = imgs[:8]

    grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
    save_image(grid, os.path.join(cfg.runtime_cfg.save_dir, f"generated_img_step{step}_cfg{cfg_scale}_denoise{denoise_timesteps}.png"))
    wandb.log({"Generated samples": wandb.Image(grid)})
    # Optionally save raw arrays for later analysis
    #if cfg.runtime_cfg.save_dir is not None:
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "x0.npy"), np.concatenate(all_x0, axis=0))
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "x1.npy"), np.concatenate(all_x1, axis=0))
    #    np.save(os.path.join(cfg.runtime_cfg.save_dir, "labels.npy"), np.concatenate(all_labels, axis=0))

    # restore mode
    ema_model.train() if was_training else ema_model.eval()


@torch.no_grad()
def inference(
    cfg,
    ema_model,                   # model on the correct device
    vae=None,                    # optional StableVAE wrapper
    num_generations=50_000,        # TOTAL images across all GPUs
    fid_stats_path=None,         # path to precomputed REAL stats (required for FID)
):
    """
    Multi-GPU FID inference. Call on ALL ranks (DDP). If DDP isn't initialized,
    it runs single-GPU on the current process.
    Assumes model forward signature: m(x_chw, t, k, y, train=False)
    and that sampling produces BCHW in [-1,1] before optional VAE decode.
    """

    # -------- DDP env --------
    is_dist = dist.is_available() and dist.is_initialized()
    world   = dist.get_world_size() if is_dist else 1
    rank    = dist.get_rank() if is_dist else 0

    # Sync everyone before heavy work
    if is_dist: dist.barrier()

    # -------- Build a small iterator only to get shape --------
    # Use per-rank batch (avoid the global/per-rank confusion)
    per_rank_bs = max(1, cfg.runtime_cfg.batch_size // world) if is_dist else cfg.runtime_cfg.batch_size
    dataset_iter = get_dataset_iter(
        cfg.runtime_cfg.dataset_name,
        cfg.runtime_cfg.dataset_root_dir,
        per_rank_bs, True, cfg.runtime_cfg.debug_overfit
    )

    batch_images, batch_labels = next(dataset_iter)
    if cfg.model_cfg.use_stable_vae:
        batch_images = vae.encode(batch_images)          # keep BHWC
    images_shape = batch_images.shape                    # [B,H,W,C] or [B,C,H,W] (we'll preserve BHWC for model)
    device = batch_images.device
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
    fid = None
    if fid_stats_path is not None:
        # --- build FID (version-safe) ---
        try:
            fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    input_img_size=(3, fid_image_size, fid_image_size),
                    antialias=True,
                    sync_on_compute=False,      # works on newer TM
                    ).to(device)
        except TypeError:
            # older TorchMetrics: no sync_on_compute arg in __init__
            fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    input_img_size=(3, fid_image_size, fid_image_size),
                    antialias=True,
                    ).to(device)
            if hasattr(fid, "sync_on_compute"):
                fid.sync_on_compute = False  # disable internal DDP sync

        d = torch.load(fid_stats_path, map_location=device)
        # Load REAL side only
        fid.real_features_sum.copy_(d["real_features_sum"].to(device))
        fid.real_features_cov_sum.copy_(d["real_features_cov_sum"].to(device))
        fid.real_features_num_samples.copy_(d["real_features_num_samples"].to(device))

    # -------- Helpers --------
    def call_model(x_bchw, t_vec, k, labels):
        return ema_model(x_bchw, t_vec, k, labels, train=False)

    #T = int(cfg.model_cfg.denoise_timesteps)
    #K = int(math.log2(T))  # max level
    denoise_T = cfg.runtime_cfg.inference_timesteps
    K = int(math.log2(denoise_T))
    dt = 1.0 / denoise_T
    cfg_scale = cfg.runtime_cfg.inference_cfg_scale

    k = torch.full((B,), float(K), device=device, dtype=torch.float32)  # per-sample level code (sentinel)

    gen = torch.Generator(device=device).manual_seed(1234 + rank)

    # Collect a tiny preview on rank 0
    preview = []

    # -------- Sampling loop --------
    generated = 0
    pbar = tqdm.tqdm(range(iters), disable=(rank != 0))
    for _ in pbar:
        take = min(B, n_local - generated)
        if take <= 0:
            break

        # fresh labels & noise
        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long, generator=gen)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes if cfg.runtime_cfg.num_classes > 1 else 0)
        x = torch.randn(images_shape, device=device, generator=gen)  # keep BHWC shape

        # Euler sampling with CFG
        for ti in range(denoise_T):
            t = ti / denoise_T
            t_vec = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 0:
                v = call_model(x, t_vec, k, labels_uncond)
            elif cfg_scale == 1:
                v = call_model(x, t_vec, k, labels)
            else:
                v_un = call_model(x, t_vec, k, labels_uncond)
                v_co = call_model(x, t_vec, k, labels)
                v = v_un + cfg_scale * (v_co - v_un)

            x = x + v * dt

        # to [0,1] and CHW for FID
        x1 = x[:take].detach()
        if cfg.model_cfg.use_stable_vae:
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)                    # BCHW in [-1,1]
        x1 = (x_vis.clamp(-1, 1) + 1.0) * 0.5        # BCHW in [0,1]

        if fid is not None:
            fid.update(x1, real=False)

        if rank == 0 and len(preview) < 8:
            need = 8 - len(preview)
            preview.extend([img for img in x1[:need]])

        generated += take
        if rank == 0:
            pbar.set_postfix_str(f"gen={generated}/{n_local}")

    # -------- Reduce FID fake stats across ranks --------
    fid_score = None
    if fid is not None:
        if is_dist:
            for name in ["fake_features_sum", "fake_features_cov_sum", "fake_features_num_samples"]:
                dist.all_reduce(getattr(fid, name), op=dist.ReduceOp.SUM)
        
        if (not is_dist) or rank == 0:
            torch.cuda.synchronize()
            with torch.amp.autocast('cuda', enabled=False):
                fid_score = fid.compute().item()
                print(f"FID: {fid_score:.4f}")
            torch.cuda.synchronize()
            if hasattr(wandb, "log") and wandb.run is not None:
                wandb.log({"metrics/FID": fid_score})
        
    # -------- Save a small grid (rank 0) --------
    if rank == 0 and len(preview) > 0:
        imgs = torch.stack(preview, dim=0)
        grid = make_grid(imgs, nrow=4, padding=2, normalize=False)
        os.makedirs(cfg.runtime_cfg.save_dir, exist_ok=True)
        tag = f"_FID{fid_score:.4f}" if fid_score is not None else ""
        out_path = os.path.join(
            cfg.runtime_cfg.save_dir,
            f"generated_step{getattr(cfg.runtime_cfg,'global_step',0)}_cfg{cfg_scale}_T{denoise_T}{tag}.png"
        )
        save_image(grid, out_path)
        if hasattr(wandb, "log") and wandb.run is not None:
            cap = f"FID: {fid_score:.4f}" if fid_score is not None else f"Samples cfg={cfg_scale} T={denoise_T}"
            wandb.log({"Generated samples": wandb.Image(grid, caption=cap)})

    # restore mode
    ema_model.train() if was_training else ema_model.eval()

    return fid_score

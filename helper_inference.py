import os, math
import wandb
import torch
import torch.distributed as dist
import tqdm
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchmetrics.image import FrechetInceptionDistance

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def do_inference(
    cfg,
    ema_model,                # torch.nn.Module or None
    dataset_iter,
    vae=None,
    num_generations=8,        # total across all GPUs (if distributed is active)
    calc_fid=False,
    step=0,
    use_distributed=False,    # True => attempt multi-GPU; falls back safely if only rank 0 calls
):
    fid_score = None

    # ---- DDP env + safe activation ----
    world = 1
    rank = 0
    is_dist = False
    if use_distributed and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        rank = dist.get_rank()
        # Try a short monitored barrier to confirm all ranks entered; otherwise fall back to single-rank.
        try:
            import datetime
            timeout_s = float(os.environ.get("INFER_BARRIER_TIMEOUT_S", "5"))
            dist.monitored_barrier(timeout=datetime.timedelta(seconds=timeout_s))
            is_dist = (world > 1)
        except Exception:
            # Only rank 0 prints the fallback notice to avoid spam
            if rank == 0:
                print("[inference] Other ranks didn't join within timeout; "
                      "falling back to single-GPU evaluation on rank 0.")
            world, rank, is_dist = 1, 0, False

    # One batch to get shape and device
    batch_images, batch_labels = next(dataset_iter)
    if cfg.model_cfg.use_stable_vae:
        batch_images = vae.encode(batch_images)

    images_shape = batch_images.shape    # [B, C, H, W]
    device = batch_images.device
    B = images_shape[0]

    # Generation budget
    n_total = int(num_generations)
    n_local = (n_total + world - 1) // world if is_dist else n_total
    iters_local = (n_local + B - 1) // B
    if rank == 0:
        print(f"[inference] world={world} | total={n_total} | per-rank={n_local} | "
              f"T={cfg.runtime_cfg.inference_timesteps} | cfg={cfg.runtime_cfg.inference_cfg_scale}")

    # EMA mode
    was_training_ema = (ema_model.training if ema_model is not None else None)
    if ema_model is not None:
        ema_model.eval()

    # ----- FID (optional) -----
    fid = None
    if (cfg.runtime_cfg.fid_stats is not None) and calc_fid:
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,                # inputs must be in [0,1]
            input_img_size=(3, 256, 256),
            antialias=True,
        ).to(device)
        d = torch.load(cfg.runtime_cfg.fid_stats, map_location=device)
        fid.real_features_sum.copy_(d["real_features_sum"].to(device))
        fid.real_features_cov_sum.copy_(d["real_features_cov_sum"].to(device))
        fid.real_features_num_samples.copy_(d["real_features_num_samples"].to(device))

    # Wrapper -> EMA
    def call_model(x, t_vector, dt_base, labels):
        m = ema_model
        return m(x, t_vector, dt_base, labels, train=False, return_activations=True)

    # dt token
    dt_flow = int(math.log2(cfg.model_cfg.denoise_timesteps))
    dt_base = torch.full((B,), dt_flow, device=device, dtype=torch.float32)

    # Rank-distinct RNG for fresh noise
    gen = torch.Generator(device=device).manual_seed(1234 + rank)

    # Sampler params
    denoise_timesteps = cfg.runtime_cfg.inference_timesteps
    cfg_scale = cfg.runtime_cfg.inference_cfg_scale
    delta_t = 1.0 / denoise_timesteps

    # For preview (rank 0 only)
    imgs_2_vis = []
    generated = 0

    pbar = tqdm.tqdm(range(iters_local), disable=(rank != 0))
    for _ in pbar:
        take = min(B, n_local - generated)
        if take <= 0:
            break

        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device,
                               dtype=torch.long, generator=gen)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes)

        x = torch.randn(images_shape, device=device, generator=gen)

        for ti in range(denoise_timesteps):
            t = (ti + 0.5) / denoise_timesteps
            t_vector = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 1:
                v = call_model(x, t_vector, dt_base, labels)
            elif cfg_scale == 0:
                v = call_model(x, t_vector, dt_base, labels_uncond)
            else:
                v_uncond = call_model(x, t_vector, dt_base, labels_uncond)
                v_cond = call_model(x, t_vector, dt_base, labels)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + v * delta_t

        # Map to [0,1] if using VAE decode
        x1 = x.detach()
        if cfg.model_cfg.use_stable_vae:
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)
            x_vis = x_vis.clamp(-1.0, 1.0)
            x1 = (x_vis + 1.0) * 0.5   # CHW in [0,1]

        # Slice last partial batch
        x1 = x1[:take]

        if fid is not None:
            fid.update(x1, real=False)

        if rank == 0 and len(imgs_2_vis) < 8:
            needed = 8 - len(imgs_2_vis)
            imgs_2_vis.extend([img for img in x1[:needed]])

        generated += take
        if rank == 0:
            pbar.set_postfix_str(f"gen={generated}/{n_local}")

    # ---- Reduce FID across ranks (only if truly distributed) ----
    if fid is not None and is_dist:
        for name in ["fake_features_sum", "fake_features_cov_sum", "fake_features_num_samples"]:
            dist.all_reduce(getattr(fid, name), op=dist.ReduceOp.SUM)

    # ---- Compute/log FID (rank 0 or single-GPU) ----
    if fid is not None and ((not is_dist) or rank == 0):
        with torch.cuda.amp.autocast(enabled=False):
            fid_score = fid.compute().item()
        print(f"[FID] {fid_score:.4f}  (N={n_total if is_dist else generated})")
        if hasattr(wandb, "log") and wandb.run is not None:
            wandb.log({"metrics/FID": fid_score}, step=step)

    # ---- Preview grid (rank 0 only) ----
    if rank == 0 and len(imgs_2_vis) > 0:
        imgs = torch.stack(imgs_2_vis, dim=0)[:8]  # [<=8, C, H, W]
        grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
        os.makedirs(cfg.runtime_cfg.save_dir, exist_ok=True)

        if fid_score is not None:
            caption = f"FID: {fid_score:.4f}"
            fname_tag = f"_FID{fid_score:.4f}"
        else:
            caption = f"Samples | cfg={cfg.runtime_cfg.inference_cfg_scale} T={cfg.runtime_cfg.inference_timesteps}"
            fname_tag = ""

        out_path = os.path.join(
            cfg.runtime_cfg.save_dir,
            f"generated_img_step{step}_cfg{cfg.runtime_cfg.inference_cfg_scale}"
            f"_denoise{cfg.runtime_cfg.inference_timesteps}{fname_tag}.png"
        )
        save_image(grid, out_path)
        if hasattr(wandb, "log") and wandb.run is not None:
            wandb.log({"Generated samples": wandb.Image(grid, caption=caption)}, step=step)

    # Restore EMA mode
    if ema_model is not None:
        ema_model.train() if was_training_ema else ema_model.eval()

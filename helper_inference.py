import wandb
import torch
import torch.distributed as dist
import math, os, tqdm
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
    num_generations=8,        # total across all GPUs
    calc_fid=False,
    step=0,
    use_distributed=False
):

    fid_score = None

    # ---- DDP env ----
    is_dist = use_distributed and dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    # One batch to get shape and device
    batch_images, batch_labels = next(dataset_iter)
    if cfg.model_cfg.use_stable_vae:
        batch_images = vae.encode(batch_images)

    images_shape = batch_images.shape          # [B, C, H, W] or your BHWC (you decode later)
    device = batch_images.device
    B      = images_shape[0]

    # Per-rank generation budget (ceil shard)
    n_total = int(num_generations)
    n_local = (n_total + world - 1) // world
    start   = rank * n_local
    end     = min(n_total, (rank + 1) * n_local)
    n_local = max(0, end - start)

    denoise_timesteps = cfg.runtime_cfg.inference_timesteps
    cfg_scale         = cfg.runtime_cfg.inference_cfg_scale
    delta_t           = 1.0 / denoise_timesteps
    if rank == 0:
        print(f"[inference] world={world} → total={n_total}, per-rank≈{n_local}, "
              f"T={denoise_timesteps}, cfg={cfg_scale}")

    # Put EMA in eval
    was_training_ema = (ema_model.training if ema_model is not None else None)
    if ema_model is not None:
        ema_model.eval()

    # ----- FID (optional) -----
    fid = None
    if (cfg.runtime_cfg.fid_stats is not None) and calc_fid:
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,              # inputs must be in [0,1]
            input_img_size=(3, 256, 256),
            antialias=True,
        ).to(device)

        # Load REAL stats (same on all ranks is fine)
        d = torch.load(cfg.runtime_cfg.fid_stats, map_location=device)
        fid.real_features_sum.copy_(d["real_features_sum"].to(device))
        fid.real_features_cov_sum.copy_(d["real_features_cov_sum"].to(device))
        fid.real_features_num_samples.copy_(d["real_features_num_samples"].to(device))

    # Wrapper that always uses EMA if provided
    def call_model(x, t_vector, dt_base, labels):
        m = ema_model
        # model forward expects BHWC in your DiT (you already use it that way)
        v_pred = m(x, t_vector, dt_base, labels, train=False, return_activations=True)
        return v_pred

    # dt token
    dt_flow = int(math.log2(cfg.model_cfg.denoise_timesteps))
    dt_base = torch.full((B,), dt_flow, device=device, dtype=torch.float32)

    # Rank-distinct RNG for fresh noise
    gen = torch.Generator(device=device).manual_seed(1234 + rank)

    # For quick visualization (rank 0 only)
    nimgs_2_vis, imgs_2_vis = 0, []

    # How many batches this rank needs to generate
    iters_local = max(n_local // B, 0)
    if n_local > 0 and (n_local % B) != 0:
        iters_local += 1  # last partial batch is OK; we’ll slice after

    for _ in tqdm.tqdm(range(iters_local), disable=(rank != 0)):
        # Labels per rank
        labels = torch.randint(0, cfg.runtime_cfg.num_classes, (B,), device=device, dtype=torch.long, generator=gen)
        labels_uncond = torch.full_like(labels, cfg.runtime_cfg.num_classes)

        # Fresh noise per batch, per rank
        x = torch.randn(images_shape, device=device, generator=gen)

        # Simple Euler sampler (your code)
        for ti in range(denoise_timesteps):
            t = (ti + 0.5) / denoise_timesteps
            t_vector = torch.full((B,), t, device=device, dtype=torch.float32)

            if cfg_scale == 1:
                v = call_model(x, t_vector, dt_base, labels)
            elif cfg_scale == 0:
                v = call_model(x, t_vector, dt_base, labels_uncond)
            else:
                v_uncond = call_model(x, t_vector, dt_base, labels_uncond)
                v_cond   = call_model(x, t_vector, dt_base, labels)
                v        = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + v * delta_t

        # Map to [0,1] if using VAE decode
        x1 = x.detach()
        if cfg.model_cfg.use_stable_vae:
            with torch.inference_mode(), torch.amp.autocast('cuda', torch.float16):
                x_vis = vae.decode(x1)
            x_vis = x_vis.clamp(-1.0, 1.0)
            x1 = (x_vis + 1.0) * 0.5   # CHW in [0,1]

        # Clip to the number we actually need on this last iter
        remaining = n_local - (len(imgs_2_vis) * B) if rank == 0 else n_local  # only used for FID slice below
        if remaining < B and remaining > 0:
            x1 = x1[:remaining]

        if fid is not None:
            fid.update(x1, real=False)

        if rank == 0 and nimgs_2_vis < 8:
            imgs_2_vis.append(x1)
            nimgs_2_vis += x1.shape[0]

    # ----- Reduce FID across ranks and compute on rank 0 -----
    if fid is not None and is_dist:
        for name in ["fake_features_sum", "fake_features_cov_sum", "fake_features_num_samples"]:
            t = getattr(fid, name)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    if fid is not None and ((not is_dist) or rank == 0):
        with torch.cuda.amp.autocast(enabled=False):
            fid_score = fid.compute().item()
        print(f"[rank0] FID = {fid_score:.4f}  (N={n_total})")
        if hasattr(wandb, "log") and wandb.run is not None:
            wandb.log({"metrics/FID": fid_score}, step=step)

    # ---- Save a grid only from rank 0 (avoid duplicates) ----
    if rank == 0 and len(imgs_2_vis) > 0:
        imgs = torch.cat(imgs_2_vis, dim=0)[:8]
        grid = vutils.make_grid(imgs, nrow=4, padding=2, normalize=False)
        os.makedirs(cfg.runtime_cfg.save_dir, exist_ok=True)

        # Title/caption text for W&B image
        if fid_score is not None:
            caption = f"FID: {fid_score:.4f}"
            fname_tag = f"_FID{fid_score:.4f}"
        else:
            caption = f"Samples (no FID) | cfg={cfg.runtime_cfg.inference_cfg_scale} T={cfg.runtime_cfg.inference_timesteps}"
            fname_tag = ""

        out_path = os.path.join(
            cfg.runtime_cfg.save_dir,
            f"generated_img_step{step}_cfg{cfg.runtime_cfg.inference_cfg_scale}_denoise{cfg.runtime_cfg.inference_timesteps}{fname_tag}.png"
        )
        save_image(grid, out_path)

        if hasattr(wandb, "log") and wandb.run is not None:
            wandb.log({"Generated samples": wandb.Image(grid, caption=caption)}, step=step)

    # Restore EMA mode
    if ema_model is not None:
        if was_training_ema:
            ema_model.train()
        else:
            ema_model.eval()

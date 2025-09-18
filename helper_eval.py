# helper_eval_torch.py
import math
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def eval_model(
    FLAGS,
    save_dir,
    model,                   # torch.nn.Module (possibly DDP-wrapped)
    ema_model,               # torch.nn.Module or None
    step: int,
    dataset_iter,            # iterator yielding (images_bhwc, labels)
    dataset_valid_iter,      # iterator yielding (images_bhwc, labels)
    vae_encode=None,         # callable or None (expects BHWC -> latents (BCHW))
    vae_decode=None,         # callable or None (expects latents (BCHW) -> BHWC)
    update_fn=None,          # callable(train_batch, labels, force_t, force_dt) -> (loss_info)
    get_fid_activations=None,# callable(images_bhwc)-> (N,2048) activations
    imagenet_labels=None,    # list[str]
    visualize_labels=None,   # not required here, kept for parity
    fid_from_stats=None,
    truth_fid_stats=None,
):
    device = next(model.parameters()).device
    model.eval()
    if ema_model is not None:
        ema_model.eval()

    # Sample one batch for shapes
    batch_images, batch_labels = next(dataset_iter)
    valid_images, valid_labels = next(dataset_valid_iter)

    # VAE encode to latents if needed (your JAX path does this before eval) :contentReference[oaicite:2]{index=2}
    if FLAGS.model['use_stable_vae'] and 'latent' not in FLAGS.dataset_name:
        if vae_encode is None:
            raise RuntimeError("vae_encode required when use_stable_vae=1")
        batch_images = vae_encode(batch_images)
        valid_images = vae_encode(valid_images)

    # If dataset already provides latent pairs, split them like JAX (eps|img) :contentReference[oaicite:3]{index=3}
    if 'latent' in FLAGS.dataset_name:
        # Follow your file’s intent; the original had slicing typos in snippet,
        # but the idea is (eps | x) along channel-last axis.
        half = valid_images.shape[-1] // 2
        eps_valid = valid_images[..., :half]
        batch_images = batch_images[..., half:]
        valid_images = valid_images[..., half:]

    # Helper to render BHWC in [0,1], optionally via VAE decode :contentReference[oaicite:4]{index=4}
    def process_img(img_bhwc):
        x = img_bhwc
        if FLAGS.model['use_stable_vae'] and vae_decode is not None:
            x = vae_decode(x)  # decode latents to image space [-1,1]
        x = (x * 0.5 + 0.5).clamp(0, 1)
        return x

    # ---- Per-t loss sweep like your JAX plots (discrete dt buckets) ---- :contentReference[oaicite:5]{index=5}
    if FLAGS.model['denoise_timesteps'] == 128:
        rows, cols = 5, 8
        d_list = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        rows, cols = 3, 6
        d_list = [0, 1, 2, 3, 4, 5]
    fig, axs = plt.subplots(rows, cols, figsize=(15, 12 if rows == 5 else 8))

    # We probe losses by forcing t in {0, 1/32, ..., 31/32} just like the JAX loop. :contentReference[oaicite:6]{index=6}
    for idx, d in enumerate(d_list):
        losses = []
        for k in range(32):
            t_forced = k / 32.0
            # pull fresh batch (JAX also re-fetches each t) :contentReference[oaicite:7]{index=7}
            try:
                bi, bl = next(dataset_iter)
            except StopIteration:
                break
            if FLAGS.model['use_stable_vae'] and 'latent' not in FLAGS.dataset_name and vae_encode is not None:
                bi = vae_encode(bi)
            # call your Torch update_fn in eval mode with force_t
            if update_fn is None:
                # fallback: forward-only loss if update_fn not provided
                loss = torch.tensor(0.0, device=device)
            else:
                # update_fn should return (loss, info) or (state, info). Accept either.
                out = update_fn(bi, bl, force_t=t_forced, force_dt=d)
                if isinstance(out, tuple) and len(out) == 2:
                    loss_info = out[1]
                elif isinstance(out, dict):
                    loss_info = out
                else:
                    # if (state, info)
                    loss_info = out[-1] if isinstance(out, (list, tuple)) else {}
                loss = torch.as_tensor(loss_info.get('loss', 0.0), device=device)
            losses.append(float(loss))
        r, c = divmod(idx, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        ax.plot(np.arange(len(losses)) / 32.0, losses)
        ax.set_title(f"dt_base={d}")
        ax.set_xlabel("t")
        ax.set_ylabel("loss")

    plt.tight_layout()
    if getattr(FLAGS, "save_dir", None):
        fig.savefig(f"{save_dir}/eval_loss_by_t.png", dpi=160)
    plt.close(fig)

    # (Optional) quick FID check for one small batch of decoded samples
    if get_fid_activations and truth_fid_stats is not None:
        # use the valid batch, decode to images in [0,1] and get acts
        x_vis = process_img(valid_images)
        acts = get_fid_activations(x_vis)
        mu, sigma = acts.mean(0).cpu().numpy(), np.cov(acts.cpu().numpy(), rowvar=False)
        fid = fid_from_stats(mu, sigma, truth_fid_stats['mu'], truth_fid_stats['sigma'])
        # simple print/log; your JAX does W&B logging elsewhere
        print(f"[eval] FID (batch approx): {fid:.2f}")

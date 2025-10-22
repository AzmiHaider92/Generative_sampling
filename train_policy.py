import math
import torch.distributed as dist
from dataclasses import asdict
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
from papers_e2e.policy_helper import get_targets
import wandb
from models.meta_model import DiT
from models.model_config import get_dit_params
from arguments_parser import RuntimeCfg, ModelCfg, WandbCfg, load_configs_from_file, parse_args, CFG
import importlib
import time
from typing import Optional
from models.policy import PolicyWithDiTEmbedders
from papers_e2e.policy_helper import sample_t
from train import print_gpu_info, set_seed, maybe_wrap_ddp, unwrap, make_ema
from utils.datasets import get_dataset as get_dataset_iter
from utils.sharding import ddp_setup
from utils.stable_vae import StableVAE
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
# At the very top of train.py (before DataLoaders are created)
import torch, os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
try:
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set


# ---------------- Train ----------------
def main():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
    except Exception:
        # older PyTorch fallback
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    is_ddp, device, rank, world, local_rank = ddp_setup()
    print_gpu_info(rank, world, local_rank)

    #device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    # defaults from code
    runtime_cfg = RuntimeCfg()
    model_cfg = ModelCfg()
    wandb_cfg = WandbCfg()

    # override from file if provided
    if args.config:
        runtime_cfg, model_cfg, wandb_cfg = load_configs_from_file(args.config, runtime_cfg, model_cfg, wandb_cfg)

    set_seed(runtime_cfg.seed)

    # ----- data -----
    # Iterator that yields (images_bhwc, labels)
    per_rank_bs = max(1, runtime_cfg.batch_size // max(1, world))
    train_iter = get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir, per_rank_bs, True, runtime_cfg.debug_overfit)
    valid_iter = get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir, per_rank_bs, False, runtime_cfg.debug_overfit)
    example_images, example_labels = next(train_iter)

    # ----- wandb -----
    if (not is_ddp) or rank == 0:
        # timestamped run name
        ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
        base_name = wandb_cfg.name.format(train_type=model_cfg.train_type, dataset_name=runtime_cfg.dataset_name)
        run_name = f"{base_name}_{ts}"
        runtime_cfg.save_dir = os.path.join(runtime_cfg.save_dir, run_name)
        os.makedirs(runtime_cfg.save_dir, exist_ok=True)

        wandb.init(
            project=wandb_cfg.project,
            name=run_name,
            id=None if wandb_cfg.run_id == "None" else wandb_cfg.run_id,
            resume="allow" if wandb_cfg.run_id != "None" else None,
            mode=wandb_cfg.mode,
            config={
                "runtime": asdict(runtime_cfg),
                "model": asdict(model_cfg),
                "wandb": asdict(wandb_cfg),
                "batch_size_total": runtime_cfg.batch_size,
                "batch_size_per_rank": per_rank_bs},
        )

    # ----- VAE (optional) -----
    vae = None
    if model_cfg.use_stable_vae:
        vae = StableVAE(device)

    def maybe_encode(x_bchw):
        if model_cfg.use_stable_vae:
            return vae.encode(x_bchw)
        return x_bchw

    # one pass to know channels
    example_images = maybe_encode(example_images)
    C, H = example_images.shape[1], example_images.shape[-1]

    # ----- model -----
    params = get_dit_params(model_id=model_cfg.model_id)
    dit = DiT(
        **params,
        in_channels=C,
        num_classes=runtime_cfg.num_classes,
        mlp_ratio=model_cfg.mlp_ratio,
        ignore_k=not (model_cfg.train_type.startswith("shortcut")),
        image_size=H,
    ).to(device)

    # --- build optional selector ---
    policy = PolicyWithDiTEmbedders(
            x_embedder=dit.x_embedder,  # reuse
            pose_embed=dit.pos_embed,
            t_embedder=dit.t_embedder,  # reuse
            y_embedder=dit.y_embedder,  # reuse
            T=model_cfg.denoise_timesteps,
        ).to(device)

    # --- wrap models with (optional) DDP ---
    dit = maybe_wrap_ddp(dit, device, is_ddp)
    policy = maybe_wrap_ddp(policy, device, is_ddp)

    # Pointers to the underlying modules (use these for EMA/source-of-truth)
    live_model = unwrap(dit, is_ddp)
    live_policy = unwrap(policy, is_ddp)

    # --- EMA setup ---
    ema_model = None
    ema_policy = None
    EMA_DECAY = 0.9999
    ema_t = None  # step counter for bias-correction if you use it later

    if model_cfg.use_ema:
        ema_model = make_ema(live_model)
        ema_policy = make_ema(live_policy)

    opt = torch.optim.AdamW(policy.head.parameters(), lr=model_cfg.lr_policy, weight_decay=0.05)

    # ----- model wrappers -----
    def _forward_model(m, x_t, t, k, labels, train=True):
        dev = next(m.parameters()).device
        # move inputs to the model's device
        x_t = x_t.to(dev, non_blocking=True)
        t = t.to(dev, dtype=torch.float32, non_blocking=True)
        k = k.to(dev, dtype=torch.float32, non_blocking=True)
        labels = labels.to(dev, dtype=torch.long, non_blocking=True)
        v_pred = m(x_t, t, k, labels, train=train)
        return v_pred

    def call_model(x_t, t, k, labels, use_ema: bool = False, train: bool = True):
        m = ema_model if use_ema else (dit.module if is_ddp else dit)
        return _forward_model(m, x_t, t, k, labels, train)

    # @torch.no_grad()
    # def call_model_teacher(x_t, t, k, labels):
    #    m = teacher_model if (teacher_model is not None) else ema_model
    #    return _forward_model(m, x_t, t, k, labels)

    # @torch.no_grad()
    # def call_model_student_ema(x_t, t, k, labels):
    #    return _forward_model(ema_model, x_t, t, k, labels)

    cfg = CFG(runtime_cfg=runtime_cfg, model_cfg=model_cfg, wandb_cfg=wandb_cfg)

    # ----- training loop -----
    gen = torch.Generator(device=device).manual_seed(runtime_cfg.seed)

    ################################################################################################################
    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #
    ################################################################################################################
    if rank == 0:  # only rank 0 prints
        print("==========================================================================")
        print("====================== start training loop ===============================")
        print("==========================================================================")

    pbar = tqdm(total=runtime_cfg.max_steps + 1, dynamic_ncols=True)
    step, max_steps = 1, runtime_cfg.max_steps + 1  # e.g., 8001
    amp_dtype = torch.bfloat16  # match original-style BF16

    while step < max_steps:
        t0 = time.time()

        batch_images, batch_labels = next(train_iter)
        batch_images = maybe_encode(batch_images)

        opt.zero_grad(set_to_none=True)
        # generate xt, t
        x_t, t = sample_t(cfg, gen, batch_images)
        # policy chooses the next dt
        dt = policy(x_t, t, batch_labels)
        # calc LTE
        x_student, xh, x2h, _, _, _, _ = get_targets(cfg,
                                                     batch_labels,
                                                     call_model,
                                                     x_t,
                                                     t,
                                                     dt)
        loss = (x_student - x2h).float().pow(2).mean(dim=(1, 2, 3))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.head.parameters(), 1.0)
        opt.step()

        if (step % runtime_cfg.log_interval) == 0:
            print(f"loss = {loss.item()}")

        step_time = time.time() - t0
        if rank == 0:
            pbar.set_postfix_str(f"step={step_time * 1e3:.0f}ms")
            pbar.update(1)

        step += 1


if __name__ == "__main__":
    main()
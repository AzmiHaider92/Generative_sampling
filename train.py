import math
from dataclasses import asdict
from datetime import datetime
import copy
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from arguments_parser import RuntimeCfg, ModelCfg, WandbCfg, load_configs_from_file, parse_args, CFG
from model import DiT
import importlib
import time
from utils.datasets import get_dataset as get_dataset_iter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.sharding import ddp_setup
from utils.stable_vae import StableVAE
from helper_inference import do_inference
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
# At the very top of train.py (before DataLoaders are created)
import torch, multiprocessing as mp, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
try:
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set


# ---------------- Utils ----------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class WarmupCosine:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = max(total_steps, warmup_steps + 1)

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * step / max(1, self.warmup)
        progress = (step - self.warmup) / (self.total - self.warmup)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))


class LinearWarmup:
    def __init__(self, base_lr: float, warmup_steps: int):
        self.base_lr = base_lr
        self.warmup = warmup_steps
    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * step / max(1, self.warmup)
        return self.base_lr


def print_gpu_info(rank=0, world_size=1, local_rank=0):
    if rank == 0:  # only rank 0 prints
        print("===================================")
        print(f"🌍 WORLD_SIZE = {world_size}")
        print(f"🖥️  Visible GPUs = {torch.cuda.device_count()}")
    print(f"[Rank {rank}/{world_size}] -> using cuda:{local_rank}")
    print("===================================")


def ddp_mean_scalar(x: float, device):
    t = torch.tensor([x], device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


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

        # helpers to keep BHWC IO for the model
        @torch.no_grad()
        def vae_encode_bhwc(x_bhwc: torch.Tensor) -> torch.Tensor:
            # returns BHWC latents
            z_bchw = vae.encode(x_bhwc)                 # BCHW (latents)
            return z_bchw.permute(0, 2, 3, 1).contiguous()

        @torch.no_grad()
        def vae_decode_bhwc(lat_bhwc: torch.Tensor) -> torch.Tensor:
            z_bchw = lat_bhwc.permute(0, 3, 1, 2).contiguous()
            x = vae.decode(z_bchw)                      # BHWC in [-1,1]
            return x
    else:
        vae_encode_bhwc = vae_decode_bhwc = None

    def maybe_encode(x_bhwc):
        if model_cfg.use_stable_vae and vae_encode_bhwc is not None and 'latent' not in runtime_cfg.dataset_name:
            return vae_encode_bhwc(x_bhwc)
        return x_bhwc

    # one pass to know channels
    example_images = maybe_encode(example_images)
    H, C = example_images.shape[1], example_images.shape[-1]

    # ----- model -----
    dit = DiT(
        in_channels=C,
        patch_size=model_cfg.patch_size,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        out_channels=C,
        class_dropout_prob=model_cfg.class_dropout_prob,
        num_classes=runtime_cfg.num_classes,
        dropout=model_cfg.dropout,
        ignore_dt=False if (model_cfg.train_type in ("shortcut","livereflow")) else True,
        image_size=H,
    ).to(device)

    if is_ddp:
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )

    live_model = dit.module if is_ddp else dit
    # EMA model: deep copy + move to device
    ema_model = copy.deepcopy(live_model).to(device)
    ema_model.eval()  # optional, but typical for EMA

    teacher_model = None
    if model_cfg.train_type in ("progressive", "consistency-distillation"):
        # copy current weights
        teacher_model = copy.deepcopy(live_model).to(device)
        teacher_model.load_state_dict((dit.module if is_ddp else dit).state_dict())

    # ----- opt + sched -----
    def param_groups(m: nn.Module, wd: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad: continue
            if p.ndim == 1 or n.endswith(".bias"): no_decay.append(p)
            else: decay.append(p)
        return [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]
    base_lr = model_cfg.lr
    lr_sched = (WarmupCosine(base_lr, model_cfg.warmup, runtime_cfg.max_steps)
                if model_cfg.use_cosine else
                (LinearWarmup(base_lr, model_cfg.warmup) if model_cfg.warmup > 0 else (lambda s: base_lr)))
    opt = torch.optim.AdamW(param_groups(dit, model_cfg.weight_decay),
                            lr=base_lr, betas=(model_cfg.beta1, model_cfg.beta2),
                            eps=1e-8, fused=True)

    amp_dtype = torch.bfloat16  # or torch.float16
    scaler = GradScaler('cuda', enabled=(amp_dtype is torch.float16))

    # ----- checkpoint load -----
    global_step = 0
    if runtime_cfg.load_dir:
        ckpt_path = runtime_cfg.load_dir if runtime_cfg.load_dir.endswith(".pt") else os.path.join(runtime_cfg.load_dir, "model.pt")
        try:
            step, extra = load_checkpoint(ckpt_path, live_model, opt, ema=None, map_location=device)
            if extra and "ema_model" in extra:
                ema_model.load_state_dict(extra["ema_model"])
            else:
                ema_model.load_state_dict(live_model.state_dict())
            global_step = int(step)
            if (not is_ddp) or rank == 0:
                print(f"[load] restored step={global_step} from {ckpt_path}")
        except FileNotFoundError:
            if (not is_ddp) or rank == 0:
                print(f"[load] no checkpoint found at {ckpt_path}")

    # ----- targets dispatcher -----
    def import_targets(train_type: str):
        name = {
            "flow_matching": "baselines.targets_flow_matching",
            "shortcut": "targets_shortcut",
            "progressive": "targets_progressive",
            "consistency-distillation": "targets_consistency_distillation",
            "consistency": "targets_consistency_training",
            "livereflow": "targets_livereflow",
        }[train_type]
        return importlib.import_module(name).get_targets
    get_targets = import_targets(model_cfg.train_type)

    # ----- model wrappers -----
    @torch.no_grad()
    def _forward_model(m, x_t, t, dt_base, labels):
        dev = next(m.parameters()).device
        # move inputs to the model's device
        x_t = x_t.to(dev, non_blocking=True)
        t = t.to(dev, dtype=torch.float32, non_blocking=True)
        dt_base = dt_base.to(dev, dtype=torch.float32, non_blocking=True)
        labels = labels.to(dev, dtype=torch.long, non_blocking=True)
        v_pred, _, _ = m(x_t, t, dt_base, labels, train=False, return_activations=False)
        return v_pred

    @torch.no_grad()
    def call_model(x_t, t, dt_base, labels, use_ema: bool = False):
        m = ema_model if use_ema else (dit.module if is_ddp else dit)
        return _forward_model(m, x_t, t, dt_base, labels)

    @torch.no_grad()
    def call_model_teacher(x_t, t, dt_base, labels, use_ema: bool = True):
        m = teacher_model if (teacher_model is not None) else ema_model
        return _forward_model(m, x_t, t, dt_base, labels)

    @torch.no_grad()
    def call_model_student_ema(x_t, t, dt_base, labels, use_ema: bool = True):
        return _forward_model(ema_model, x_t, t, dt_base, labels)

    cfg = CFG(runtime_cfg=runtime_cfg, model_cfg=model_cfg, wandb_cfg=wandb_cfg)

    # ----- eval/infer early exit -----
    if runtime_cfg.mode != "train":
        # build dataset iters again (not consumed)
        dataset = get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir, per_rank_bs, True, runtime_cfg.debug_overfit)
        dataset_valid = get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir, per_rank_bs, False, runtime_cfg.debug_overfit)

        do_inference(cfg, dit.module if is_ddp else dit,
                     (dit.module if is_ddp else dit),  # use same as ema for now
                     step=global_step,
                     dataset_iter=dataset,
                     dataset_valid_iter=dataset_valid,
                     vae_encode=vae_encode_bhwc,
                     vae_decode=vae_decode_bhwc,
                     )
        return

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
    
    #pbar = tqdm(range(global_step + 1, runtime_cfg.max_steps + 1),
    #            disable=is_ddp and rank != 0, dynamic_ncols=True)

    accum_steps = 1  # set >1 if you use gradient accumulation
    pbar = tqdm(total=runtime_cfg.max_steps + 1, dynamic_ncols=True)
    ema_t = None
    EMA_EVERY = 10
    EMA_DECAY = 0.9999
    step, max_steps = 1, runtime_cfg.max_steps + 1  # e.g., 8001

    while step < max_steps:
        t0 = time.time()

        batch_images, batch_labels = next(train_iter)
        batch_images = maybe_encode(batch_images)

        # targets per train_type
        if model_cfg.train_type == 'flow_matching':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, batch_images, batch_labels)
        elif model_cfg.train_type == 'shortcut':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, call_model, batch_images, batch_labels)
        elif model_cfg.train_type == 'progressive':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, call_model_teacher, batch_images, batch_labels, step=step)
        elif model_cfg.train_type == 'consistency-distillation':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, call_model_teacher, call_model_student_ema, batch_images, batch_labels)
        elif model_cfg.train_type == 'consistency':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, call_model_student_ema, batch_images, batch_labels)
        elif model_cfg.train_type == 'livereflow':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(cfg, gen, call_model, batch_images, batch_labels)
        else:
            raise ValueError(f"Unknown train_type: {model_cfg.train_type}")

        # unconditional path if cfg_scale == 0 (match JAX)
        if model_cfg.cfg_scale == 0:
            labels_eff = torch.full_like(labels_eff, runtime_cfg.num_classes)

        # forward + loss
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=amp_dtype):
            v_pred, _, _ = (dit)(x_t, t_vec, dt_base, labels_eff, train=True, return_activations=False)
            mse = ((v_pred - v_t) ** 2).mean(dim=(1, 2, 3))
            loss = mse.mean()

        if scaler.is_enabled():  # FP16 path
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:  # BF16 / full-precision path
            loss.backward()
            opt.step()

        # EMA update
        if model_cfg.use_ema and ema_model is not None and (step % EMA_EVERY == 0):
            ema_model.update(dit.module if is_ddp else dit, decay=EMA_DECAY ** EMA_EVERY)

        # log (every log_interval)
        if ((step % runtime_cfg.log_interval) == 0):
            # cheap scalars on every rank, averaged like JAX
            train_loss_mean = ddp_mean_scalar(float(loss.detach().cpu()), device)
            vmag_mean = ddp_mean_scalar(float(v_pred.square().mean().sqrt().detach().cpu()), device)
            lr_mean = ddp_mean_scalar(float(lr_sched(step)), device)

            # quick, single-rank validation (no activations)
            if (not is_ddp) or rank == 0:
                try:
                    vimg, vlbl = next(valid_iter)
                except StopIteration:
                    valid_iter = get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir,
                                                  per_rank_bs, False, runtime_cfg.debug_overfit)
                    vimg, vlbl = next(valid_iter)

                vimg = maybe_encode(vimg)
                vimg = vimg.to(device, non_blocking=True)
                vlbl = vlbl.to(device, non_blocking=True)

                with torch.inference_mode(), autocast('cuda', dtype=amp_dtype):
                    if model_cfg.train_type in ("flow_matching", "shortcut", "livereflow"):
                        v_x_t, v_v_t, v_t_vec, v_dt, v_lbl, _ = get_targets(cfg, gen, vimg, vlbl)
                    elif model_cfg.train_type == "progressive":
                        v_x_t, v_v_t, v_t_vec, v_dt, v_lbl, _ = get_targets(cfg, gen, call_model_teacher, vimg, vlbl,
                                                                            step=step)
                    else:  # "consistency" / "consistency-distillation"
                        v_x_t, v_v_t, v_t_vec, v_dt, v_lbl, _ = get_targets(cfg, gen, call_model_student_ema, vimg,
                                                                            vlbl)

                    v_pred2, _, _ = dit(v_x_t, v_t_vec, v_dt, v_lbl, train=False, return_activations=False)
                    v_loss = ((v_pred2 - v_v_t).pow(2).mean(dim=(1, 2, 3))).mean().item()

                wandb.log({
                    "training/loss": train_loss_mean,
                    "training/v_magnitude_prime": vmag_mean,
                    "training/lr": lr_mean,
                    "training/loss_valid": v_loss,
                }, step=step)

        # stepwise LR (optional)
        for g in opt.param_groups: g['lr'] = lr_sched(step)

        # progressive: refresh teacher
        if model_cfg.train_type == 'progressive':
            num_sections = int(math.log2(model_cfg.denoise_timesteps))
            if step % max(1, (runtime_cfg.max_steps // max(1, num_sections))) == 0 and teacher_model is not None:
                teacher_model.load_state_dict((dit.module if is_ddp else dit).state_dict())

        # eval
        if (step % runtime_cfg.eval_interval) == 0 and rank == 0:
            print("================= evaluating =================")
            if (not is_ddp) or rank == 0:
                do_inference(cfg,
                             dit.module if is_ddp else dit,
                             (dit.module if is_ddp else dit) if ema_model is None else None,
                             # pass a real ema_model if you keep a separate module
                             dataset_iter=get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir,
                                                           per_rank_bs, True, runtime_cfg.debug_overfit),
                             vae_encode=vae_encode_bhwc,
                             vae_decode=vae_decode_bhwc)

        # save
        if (step % runtime_cfg.save_interval) == 0 and runtime_cfg.save_dir and rank == 0:
            print("================= saving a checkpoint =================")
            if (not is_ddp) or rank == 0:
                os.makedirs(runtime_cfg.save_dir, exist_ok=True)
                ckpt_path = os.path.join(runtime_cfg.save_dir, f"model_step_{step}.pt")
                save_checkpoint(ckpt_path, dit.module if is_ddp else dit, opt, step=step, ema=ema_model)
                print(f"[save] {ckpt_path}")

        # update bar
        step_time = time.time() - t0
        ema_t = step_time if ema_t is None else 0.9*ema_t + 0.1*step_time
        imgs_per_sec = (runtime_cfg.batch_size * accum_steps) / ema_t
        if rank == 0:
            pbar.set_postfix_str(f"GBS={runtime_cfg.batch_size} img/s={imgs_per_sec:.0f} step={ema_t*1e3:.0f}ms")
            pbar.update(1)

        step += 1

    do_inference(cfg,
                 dit.module if is_ddp else dit,
                 (dit.module if is_ddp else dit) if ema_model is None else None,
                 # pass a real ema_model if you keep a separate module
                 dataset_iter=get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir,
                                               per_rank_bs, True, runtime_cfg.debug_overfit),
                 vae_encode=vae_encode_bhwc,
                 vae_decode=vae_decode_bhwc,
                 num_generations=50000,
                 calc_fid=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

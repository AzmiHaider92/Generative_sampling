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

from DiT.dt_selector import DtSelector
from DiT.meta_model import DiT
from DiT.model_config import get_dit_params
from arguments_parser import RuntimeCfg, ModelCfg, WandbCfg, load_configs_from_file, parse_args, CFG
import importlib
import time

from utils.datasets import get_dataset as get_dataset_iter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.sharding import ddp_setup
from utils.stable_vae import StableVAE
from test import inference, validate
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


def ddp_mean(loss: torch.Tensor) -> float | None:
    x = loss.detach().float()               # stay on GPU, fp32
    if dist.is_available() and dist.is_initialized():
        dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)   # sum to rank 0
        if dist.get_rank() == 0:
            x /= dist.get_world_size()
            return x.item()
        return None
    return x.item()


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

    # ----- method= flow matching / shortcut / adaptive step (learned dt) ... ) -----
    def import_targets(train_type: str):
        name = {
            "flow_matching": "papers_e2e.flow_matching",
            "consistency": "papers_e2e.consistency",
            "shortcut": "papers_e2e.shortcut",
            "shortcut_cont": "papers_e2e.shortcut_cont",
            #"adaptive_step": "papers_e2e.adaptive_step"
        }[train_type]
        return importlib.import_module(name).get_targets
    get_targets = import_targets(model_cfg.train_type)

    # ----- wandb -----
    if (not is_ddp) or rank == 0:
        # timestamped run name
        ts = datetime.now().strftime("M%m-D%d-H%H_M%M")
        base_name = wandb_cfg.name.format(train_type=model_cfg.train_type, dt_mode=model_cfg.dt_mode, dataset_name=runtime_cfg.dataset_name)
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
        ignore_k=(model_cfg.train_type not in ("shortcut", "livereflow")),
        image_size=H,
    ).to(device)

    dt_selector = None
    if model_cfg.train_type == "adaptive_step":
        dt_selector = DtSelector(
            K=int(math.log2(model_cfg.denoise_timesteps)),
            token_dim=C,  # e.g., C if x_t is [B,C,H,W] or D for tokens
            t_dim=128, class_dim=64, num_classes=runtime_cfg.num_classes,
                                                 mode="discrete").to(device)

    if is_ddp:
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )
        if dt_selector is not None:
            dt_selector = torch.nn.parallel.DistributedDataParallel(
                dt_selector, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
            )

    live_model = dit.module if is_ddp else dit
    live_model_selector = dt_selector.module if is_ddp else dit

    # EMA model: deep copy + move to device
    ema_model = None
    ema_selector = None

    ema_t = None
    EMA_DECAY = 0.9999
    if model_cfg.use_ema:
        ema_model = copy.deepcopy(live_model).to(device)
        ema_model.eval()
        for p in ema_model.parameters(): p.data = p.data.float(); p.requires_grad_(False)

        ema_selector = copy.deepcopy(live_model_selector).to(device)
        ema_selector.eval()
        for p in ema_selector.parameters(): p.data = p.data.float(); p.requires_grad_(False)

    # ----- opt + sched -----
    base_lr = model_cfg.lr
    warmup = model_cfg.warmup
    min_lr = 0.1 * base_lr

    def param_groups(m: nn.Module, wd: float):
        decay, no_decay = [], []
        skip = ("bias", "pos_embed", "cls_token", "time_embed", "label_embed", "embed", "norm", "ln")
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or any(s in n.lower() for s in skip):
                no_decay.append(p)  # norm/embeds/bias -> no WD
            else:
                decay.append(p)  # linear/conv weights -> WD
        return [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]

    def lr_sched(step):
        # linear warmup -> cosine decay to min_lr
        if step < warmup:
            return base_lr * (step / max(1, warmup))
        t = (step - warmup) / max(1, runtime_cfg.max_steps - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

    opt = torch.optim.AdamW(param_groups(dit, wd=0.05), lr=base_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    if dt_selector is not None:
        opt_sel = torch.optim.AdamW(dt_selector.parameters(), lr=model_cfg.lr_selector, weight_decay=0.05)

    amp_dtype = torch.bfloat16  # match original-style BF16
    #    scaler = GradScaler('cuda', enabled=(amp_dtype is torch.float16))

    # ----- checkpoint load -----
    global_step = 0
    if runtime_cfg.load_dir:
        ckpt_path = runtime_cfg.load_dir if runtime_cfg.load_dir.endswith(".pt") else os.path.join(runtime_cfg.load_dir, "model.pt")
        try:
            step = load_checkpoint(ckpt_path, live_model, opt, ema=ema_model, map_location=device)
            global_step = int(step)
            if (not is_ddp) or rank == 0:
                print(f"[load] restored step={global_step} from {ckpt_path}")
        except FileNotFoundError:
            if (not is_ddp) or rank == 0:
                print(f"[load] no checkpoint found at {ckpt_path}")

    # ----- model wrappers -----
    @torch.no_grad()
    def _forward_model(m, x_t, t, k, labels, train=False):
        dev = next(m.parameters()).device
        # move inputs to the model's device
        x_t = x_t.to(dev, non_blocking=True)
        t = t.to(dev, dtype=torch.float32, non_blocking=True)
        k = k.to(dev, dtype=torch.float32, non_blocking=True)
        labels = labels.to(dev, dtype=torch.long, non_blocking=True)
        v_pred = m(x_t, t, k, labels, train=train)
        return v_pred

    @torch.no_grad()
    def call_model(x_t, t, k, labels, use_ema: bool = True, train: bool = False):
        m = ema_model if use_ema else (dit.module if is_ddp else dit)
        return _forward_model(m, x_t, t, k, labels, train)

    #@torch.no_grad()
    #def call_model_teacher(x_t, t, k, labels):
    #    m = teacher_model if (teacher_model is not None) else ema_model
    #    return _forward_model(m, x_t, t, k, labels)

    #@torch.no_grad()
    #def call_model_student_ema(x_t, t, k, labels):
    #    return _forward_model(ema_model, x_t, t, k, labels)

    cfg = CFG(runtime_cfg=runtime_cfg, model_cfg=model_cfg, wandb_cfg=wandb_cfg)

    # ----- eval/infer early exit -----
    if runtime_cfg.mode != "train":
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        inference(cfg,
                     ema_model,
                     vae=vae,
                     num_generations=runtime_cfg.inference_num_generations,
                     fid_stats_path=runtime_cfg.fid_stats)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
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

    accum_steps = 1  # set >1 if you use gradient accumulation
    pbar = tqdm(total=runtime_cfg.max_steps + 1, dynamic_ncols=True)
    step, max_steps = global_step+1, runtime_cfg.max_steps + 1  # e.g., 8001

    while step < max_steps:
        t0 = time.time()

        batch_images, batch_labels = next(train_iter)
        batch_images = maybe_encode(batch_images)

        # targets per train_type

        if model_cfg.train_type == 'flow_matching':
            x_t, v_t, t_vec, dt_vec, labels_eff, info = get_targets(cfg, gen, batch_images, batch_labels, call_model, step)
        elif model_cfg.train_type == 'consistency':
            x_t, v_t, t_vec, dt_vec, labels_eff, info = get_targets(cfg, gen, batch_images, batch_labels, call_model, step)
        elif model_cfg.train_type == 'shortcut':
            x_t, v_t, t_vec, dt_vec, labels_eff, info = get_targets(cfg, gen, batch_images, batch_labels, call_model, step)
        elif model_cfg.train_type == 'shortcut_cont':
            x_t, v_t, t_vec, dt_vec, labels_eff, info = get_targets(cfg, gen, batch_images, batch_labels, call_model, step)
        elif model_cfg.train_type == 'adaptive_step':
            x_t, v_t, t_vec, dt_vec, labels_eff, _ = get_targets(cfg, gen, batch_images, batch_labels, call_model, step,
                                                                 dt_selector=dt_selector, selector_mode="discrete",
                                                                 selector_exploration_p=0.1)
        else:
            raise ValueError(f"Unknown train_type: {model_cfg.train_type}")

        # forward + loss
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=amp_dtype):
            v_pred = (dit)(x_t, t_vec, dt_vec, labels_eff, train=True)
            mse = (v_pred - v_t).float().pow(2).mean(dim=(1, 2, 3))
            loss = mse.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
        opt.step()

        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), (dit.module if is_ddp else dit).parameters()):
                    p_ema.data.mul_(EMA_DECAY).add_(p.data.float(), alpha=1 - EMA_DECAY)  # keep EMA in fp32

        # log (every log_interval)
        lr = lr_sched(step)
        for g in opt.param_groups: g['lr'] = lr

        if (step % runtime_cfg.log_interval) == 0:
            # cheap scalars on every rank, averaged like JAX
            train_loss_mean = ddp_mean(loss)
            vmag_mean = ddp_mean(v_pred.square().mean().sqrt())

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
                    x_t, v_t, t_vec, dt_vec, v_lbl, info = get_targets(cfg, gen, vimg, vlbl, call_model, step)
                    v_pred2 = dit(x_t, t_vec, dt_vec, v_lbl, train=False)
                    v_loss = ((v_pred2 - v_t).pow(2).mean(dim=(1, 2, 3))).mean().item()

                wandb.log({
                    "training/loss": train_loss_mean,
                    "training/v_magnitude_prime": vmag_mean,
                    "training/lr": lr,
                    "training/loss_valid": v_loss,
                }, step=step)


        # progressive: refresh teacher
        #if model_cfg.train_type == 'progressive':
        #    num_sections = int(math.log2(model_cfg.denoise_timesteps))
        #    if step % max(1, (runtime_cfg.max_steps // max(1, num_sections))) == 0 and teacher_model is not None:
        #        teacher_model.load_state_dict((dit.module if is_ddp else dit).state_dict())

        # eval
        if ((step % runtime_cfg.eval_interval) == 0 or step == 1) and rank == 0 :
            print("================= evaluating =================")
            if (not is_ddp) or rank == 0:
                validate(cfg,
                             ema_model,
                             # pass a real ema_model if you keep a separate module
                             dataset_iter=get_dataset_iter(runtime_cfg.dataset_name, runtime_cfg.dataset_root_dir,
                                                           per_rank_bs, True, runtime_cfg.debug_overfit),
                             vae=vae,
                             step=step)

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

    print("===========done training===========")
    # everyone enters together
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    fid = inference(                       # or do_inference_fid_ddp(...
            cfg,
            ema_model,
            vae=vae,
            num_generations=50000,  # total across ALL GPUs
            fid_stats_path=runtime_cfg.fid_stats
            )

    # wait for all ranks to finish before tearing down
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

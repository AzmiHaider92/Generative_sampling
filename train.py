# train_torch.py
import os, math, argparse, itertools
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any, Optional, Dict, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

# ---------- our Torch ports ----------
from model import DiT
# choose one targets_*_torch file at runtime based on FLAGS.model.train_type
import importlib

from utils.train_state import EMA
from utils.datasets import get_dataset as get_dataset_iter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.sharding import ddp_setup
from utils.fid import get_fid_network, fid_from_stats
from utils.stable_vae import StableVAE

from helper_eval import eval_model
from helper_inference import do_inference


# ---------------- Configs (mirror ml_collections) ----------------
@dataclass
class ModelCfg:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    use_cosine: int = 0
    warmup: int = 0
    dropout: float = 0.0
    hidden_size: int = 64
    patch_size: int = 8
    depth: int = 2
    num_heads: int = 2
    mlp_ratio: int = 1
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    denoise_timesteps: int = 128
    cfg_scale: float = 4.0
    target_update_rate: float = 0.999
    use_ema: int = 0
    use_stable_vae: int = 1
    sharding: str = "dp"  # kept for parity; we use DDP
    t_sampling: str = "discrete-dt"
    dt_sampling: str = "uniform"
    bootstrap_cfg: int = 0
    bootstrap_every: int = 8
    bootstrap_ema: int = 1
    bootstrap_dt_bias: int = 0
    train_type: str = "shortcut"   # naive | shortcut | progressive | consistency[-distillation] | livereflow

@dataclass
class WandbCfg:
    project: str = "shortcut"
    name: str = "shortcut_{dataset_name}"
    run_id: str = "None"
    mode: str = "online"  # or "disabled"


# ---------------- Argparse ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', type=str, default='tiny-imagenet-256')
    p.add_argument('--load_dir', type=str, default=None)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--fid_stats', type=str, default=None)

    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=20000)
    p.add_argument('--save_interval', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_steps', type=int, default=1_000_000)
    p.add_argument('--debug_overfit', type=int, default=0)
    p.add_argument('--mode', type=str, choices=['train', 'inference'], default='train')

    # model overrides (keep names close to original)
    p.add_argument('--model.lr', type=float, default=None)
    p.add_argument('--model.use_cosine', type=int, default=None)
    p.add_argument('--model.warmup', type=int, default=None)
    p.add_argument('--model.dropout', type=float, default=None)
    p.add_argument('--model.hidden_size', type=int, default=None)
    p.add_argument('--model.patch_size', type=int, default=None)
    p.add_argument('--model.depth', type=int, default=None)
    p.add_argument('--model.num_heads', type=int, default=None)
    p.add_argument('--model.mlp_ratio', type=int, default=None)
    p.add_argument('--model.class_dropout_prob', type=float, default=None)
    p.add_argument('--model.num_classes', type=int, default=None)
    p.add_argument('--auto_num_classes', type=int, default=1,
                   help='If 1 and model.num_classes not explicitly set, infer from dataset_name.')
    p.add_argument('--model.denoise_timesteps', type=int, default=None)
    p.add_argument('--model.cfg_scale', type=float, default=None)
    p.add_argument('--model.target_update_rate', type=float, default=None)
    p.add_argument('--model.use_ema', type=int, default=None)
    p.add_argument('--model.use_stable_vae', type=int, default=None)
    p.add_argument('--model.bootstrap_every', type=int, default=None)
    p.add_argument('--model.bootstrap_cfg', type=int, default=None)
    p.add_argument('--model.bootstrap_ema', type=int, default=None)
    p.add_argument('--model.train_type', type=str, default=None)

    p.add_argument('--wandb.project', type=str, default=None)
    p.add_argument('--wandb.name', type=str, default=None)
    p.add_argument('--wandb.run_id', type=str, default="None")
    p.add_argument('--wandb.mode', type=str, default=None)
    return p.parse_args()


def apply_overrides(cfg: ModelCfg, args: argparse.Namespace):
    for k, v in vars(args).items():
        if v is None:  # user didn’t pass this flag → keep default
            continue
        if k.startswith('model.'):
            name = k.split('.', 1)[1]
            if hasattr(cfg, name):
                setattr(cfg, name, v)

    # optional: infer classes from dataset if user didn’t pass --model.num_classes
    if getattr(args, 'auto_num_classes', 1) and vars(args).get('model.num_classes') is None:
        name = args.dataset_name.lower()
        if name.startswith('tiny-imagenet'):
            cfg.num_classes = 200
        elif name.startswith('cifar100'):
            cfg.num_classes = 100


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


class CfgView:
    def __init__(self, cfg): object.__setattr__(self, "_cfg", cfg)
    def __getattr__(self, n): return getattr(self._cfg, n)      # dot access
    def __setattr__(self, n, v): setattr(self._cfg, n, v)
    def __getitem__(self, k): return getattr(self._cfg, k)      # dict-style
    def __setitem__(self, k, v): setattr(self._cfg, k, v)


# ---------------- Train ----------------
def main():
    is_ddp, rank, world = ddp_setup()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    model_cfg = ModelCfg()  # defaults live here
    wandb_cfg = WandbCfg()
    apply_overrides(model_cfg, args)  # overrides only if user passed flags

    FLAGS = SimpleNamespace(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        model=CfgView(model_cfg),  # targets can do FLAGS.model['foo']
    )
    set_seed(args.seed)

    # ----- data -----
    # Iterator that yields (images_bhwc, labels)
    per_rank_bs = max(1, args.batch_size // max(1, world))
    train_iter = get_dataset_iter(args.dataset_name, per_rank_bs, True, args.debug_overfit)
    valid_iter = get_dataset_iter(args.dataset_name, per_rank_bs, False, args.debug_overfit)
    example_images, example_labels = next(train_iter)
    H = example_images.shape[1]; W = example_images.shape[2]; C = example_images.shape[3]

    # ----- wandb -----
    if (not is_ddp) or rank == 0:
        run_name = (wandb_cfg.name or "run").format(dataset_name=args.dataset_name)
        wandb.init(project=wandb_cfg.project, name=run_name,
                   id=None if wandb_cfg.run_id == "None" else wandb_cfg.run_id,
                   resume="allow" if wandb_cfg.run_id != "None" else None,
                   mode=wandb_cfg.mode,
                   config={**asdict(model_cfg),
                           "dataset_name": args.dataset_name,
                           "batch_size_total": args.batch_size,
                           "batch_size_per_rank": per_rank_bs})

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

    # If we're in latent dataset mode, split channels like JAX
    if 'latent' in args.dataset_name:
        example_images = example_images[..., example_images.shape[-1] // 2:]

    # ----- FID -----
    if args.fid_stats is not None:
        get_fid_acts = get_fid_network(device=device)
        truth_fid_stats = np.load(args.fid_stats)
    else:
        get_fid_acts = None
        truth_fid_stats = None

    # ----- model -----
    dit = DiT(
        in_channels=example_images.shape[-1],
        patch_size=model_cfg.patch_size,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        out_channels=(example_images.shape[-1] if not model_cfg.use_stable_vae else vae_encode_bhwc(example_images).shape[-1]),
        class_dropout_prob=model_cfg.class_dropout_prob,
        num_classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
        ignore_dt=False if (model_cfg.train_type in ("shortcut","livereflow")) else True,
        image_size=H,
    ).to(device)

    if is_ddp:
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[device.index], output_device=device.index, find_unused_parameters=True
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
    lr_sched = (WarmupCosine(base_lr, model_cfg.warmup, args.max_steps)
                if model_cfg.use_cosine else
                (LinearWarmup(base_lr, model_cfg.warmup) if model_cfg.warmup > 0 else (lambda s: base_lr)))
    opt = torch.optim.AdamW(param_groups(dit, model_cfg.weight_decay),
                            lr=base_lr, betas=(model_cfg.beta1, model_cfg.beta2),
                            eps=1e-8, fused=True)

    amp_dtype = torch.bfloat16  # or torch.float16
    scaler = GradScaler('cuda', enabled=(amp_dtype is torch.float16))

    # ----- checkpoint load -----
    global_step = 0
    if args.load_dir:
        ckpt_path = args.load_dir if args.load_dir.endswith(".pt") else os.path.join(args.load_dir, "model.pt")
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
            "naive": "targets_naive",
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
        v_pred, _, _ = m(x_t, t, dt_base, labels, train=False, return_activations=True)
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

    # ----- eval/infer early exit -----
    if args.mode != "train":
        # build dataset iters again (not consumed)
        dataset = get_dataset_iter(args.dataset_name, per_rank_bs, True, args.debug_overfit)
        dataset_valid = get_dataset_iter(args.dataset_name, per_rank_bs, False, args.debug_overfit)

        do_inference(args, dit.module if is_ddp else dit,
                     (dit.module if is_ddp else dit),  # use same as ema for now
                     step=global_step,
                     dataset_iter=dataset,
                     dataset_valid_iter=dataset_valid,
                     vae_encode=vae_encode_bhwc,
                     vae_decode=vae_decode_bhwc,
                     get_fid_activations=get_fid_acts,
                     imagenet_labels=open('data/imagenet_labels.txt').read().splitlines() if os.path.exists('data/imagenet_labels.txt') else None,
                     visualize_labels=None,
                     fid_from_stats=fid_from_stats,
                     truth_fid_stats=truth_fid_stats)
        return

    # ----- training loop -----
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def maybe_encode(x_bhwc):
        if model_cfg.use_stable_vae and vae_encode_bhwc is not None and 'latent' not in args.dataset_name:
            return vae_encode_bhwc(x_bhwc)
        return x_bhwc

    # one pass to know channels
    example_images = maybe_encode(example_images)

    pbar = tqdm(range(global_step + 1, args.max_steps + 1),
                disable=is_ddp and rank != 0, dynamic_ncols=True)
    for i in pbar:
        # fetch batch
        try:
            batch_images, batch_labels = next(train_iter)
        except StopIteration:
            train_iter = get_dataset_iter(args.dataset_name, per_rank_bs, True, args.debug_overfit)
            batch_images, batch_labels = next(train_iter)

        batch_images = maybe_encode(batch_images)

        # targets per train_type
        if model_cfg.train_type == 'naive':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model, batch_images, batch_labels)
        elif model_cfg.train_type == 'shortcut':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model, batch_images, batch_labels)
        elif model_cfg.train_type == 'progressive':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model_teacher, batch_images, batch_labels, step=i)
        elif model_cfg.train_type == 'consistency-distillation':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model_teacher, call_model_student_ema, batch_images, batch_labels)
        elif model_cfg.train_type == 'consistency':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model_student_ema, batch_images, batch_labels)
        elif model_cfg.train_type == 'livereflow':
            x_t, v_t, t_vec, dt_base, labels_eff, info = get_targets(FLAGS, gen, call_model, batch_images, batch_labels)
        else:
            raise ValueError(f"Unknown train_type: {model_cfg.train_type}")

        # unconditional path if cfg_scale == 0 (match JAX)
        if model_cfg.cfg_scale == 0:
            labels_eff = torch.full_like(labels_eff, model_cfg.num_classes)

        # forward + loss
        opt.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=amp_dtype):
            v_pred, logvars, activ = (dit)(x_t, t_vec, dt_base, labels_eff, train=True, return_activations=True)
            mse = ((v_pred - v_t) ** 2).mean(dim=(1, 2, 3))
            loss = mse.mean()

        if scaler.is_enabled():  # FP16 path
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:  # BF16 / full-precision path
            loss.backward()
            opt.step()
        
        # debug (unused layers/params)
        #if int(os.environ.get("RANK", "0")) == 0:
        #    unused = [n for n,p in dit.named_parameters() if p.requires_grad and p.grad is None]
        #    if unused:
        #        print("UNUSED PARAMS:", unused[:20], "… count:", len(unused))

        # EMA update
        if model_cfg.use_ema and ema_model is not None:
            ema_model.update(dit.module if is_ddp else dit)

        # log (every log_interval)
        if ((i % args.log_interval) == 0) or (i == 1):
            with torch.no_grad():
                train_metrics = {
                    "training/loss": float(loss.detach().cpu()),
                    "training/v_magnitude_prime": float(v_pred.square().mean().sqrt().detach().cpu()),
                    "training/grad_norm": float(torch.nn.utils.clip_grad_norm_((dit.module if is_ddp else dit).parameters(), max_norm=float('inf')).detach().cpu()),
                    "training/param_norm": float(sum(p.data.norm().item() ** 2 for p in (dit.module if is_ddp else dit).parameters()) ** 0.5),
                    "training/lr": lr_sched(i),
                }
                # split bootstrap vs flow (for shortcut/livereflow)
                if model_cfg.train_type in ("shortcut", "livereflow"):
                    bs = args.batch_size // model_cfg.bootstrap_every
                    train_metrics["training/loss_bootstrap"] = float(mse[:bs].mean().detach().cpu())
                    train_metrics["training/loss_flow"] = float(mse[bs:].mean().detach().cpu())
                # simple valid pass (one batch)
                try:
                    vimg, vlbl = next(valid_iter)
                except StopIteration:
                    valid_iter = get_dataset_iter(args.dataset_name, per_rank_bs, False, args.debug_overfit)
                    vimg, vlbl = next(valid_iter)
                vimg = maybe_encode(vimg)
                with autocast('cuda', dtype=amp_dtype):
                    v_x_t, v_v_t, v_t_vec, v_dt, v_lbl, _ = (
                        get_targets(FLAGS, gen, call_model, vimg, vlbl)
                        if model_cfg.train_type in ("naive", "shortcut", "livereflow")
                        else (get_targets(FLAGS, gen, call_model_teacher, vimg, vlbl, step=i)
                              if model_cfg.train_type == "progressive"
                              else get_targets(FLAGS, gen, call_model_student_ema, vimg, vlbl))
                    )
                    v_pred2, _, _ = (dit)(v_x_t, v_t_vec, v_dt, v_lbl, train=False, return_activations=True)
                    v_loss = ((v_pred2 - v_v_t) ** 2).mean(dim=(1,2,3)).mean()
                train_metrics["training/loss_valid"] = float(v_loss.detach().cpu())

            if (not is_ddp) or rank == 0:
                wandb.log(train_metrics, step=i)

        # stepwise LR (optional)
        for g in opt.param_groups: g['lr'] = lr_sched(i)

        # progressive: refresh teacher
        if model_cfg.train_type == 'progressive':
            num_sections = int(math.log2(model_cfg.denoise_timesteps))
            if i % max(1, (args.max_steps // max(1, num_sections))) == 0 and teacher_model is not None:
                teacher_model.load_state_dict((dit.module if is_ddp else dit).state_dict())

        # eval
        if (i % args.eval_interval) == 0:
            eval_model(FLAGS,
                       args.save_dir,
                       dit.module if is_ddp else dit,
                       (dit.module if is_ddp else dit) if ema_model is None else None,  # pass a real ema_model if you keep a separate module
                       step=i,
                       dataset_iter=get_dataset_iter(args.dataset_name, per_rank_bs, True, args.debug_overfit),
                       dataset_valid_iter=get_dataset_iter(args.dataset_name, per_rank_bs, False, args.debug_overfit),
                       vae_encode=vae_encode_bhwc,
                       vae_decode=vae_decode_bhwc,
                       update_fn=lambda imgs, lbls, force_t=-1, force_dt=-1: {
                           "loss": float(loss.detach().cpu())
                       },  # lightweight placeholder; you can wire a true eval step if desired
                       get_fid_activations=get_fid_acts,
                       imagenet_labels=open('data/imagenet_labels.txt').read().splitlines() if os.path.exists('data/imagenet_labels.txt') else None,
                       visualize_labels=None,
                       fid_from_stats=fid_from_stats,
                       truth_fid_stats=truth_fid_stats)

        # save
        if (i % args.save_interval) == 0 and args.save_dir:
            if (not is_ddp) or rank == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"model_step_{i}.pt")
                save_checkpoint(ckpt_path, dit.module if is_ddp else dit, opt, step=i, ema=ema_model)
                print(f"[save] {ckpt_path}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
